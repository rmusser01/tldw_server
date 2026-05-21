import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render as rtlRender, screen, waitFor, within } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import {
  BuddyShellRenderContextProvider,
  useBuddyShellRenderContext
} from "@/components/Common/PersonaBuddy"
import { usePersonaVisualRuntimeStore } from "@/store/persona-visual-runtime"
import { SELECTED_ASSISTANT_STORAGE_KEY } from "@/utils/selected-assistant-storage"

const mocks = vi.hoisted(() => ({
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
    capabilities:
      | { hasPersona: boolean; hasPersonalization?: boolean }
      | null
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
  buildPersonaWebSocketUrl: vi.fn(() => "ws://persona.test/api/v1/persona/stream"),
  fetchCompanionConversationPrompts: vi.fn(),
  buddyShellContextSnapshots: [] as Array<Record<string, any> | null>
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
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
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
    (mocks.fetchCompanionConversationPrompts as (...args: unknown[]) => unknown)(
      ...args
    )
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
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
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
    }: any) => (
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

const BuddyShellContextProbe = () => {
  const context = useBuddyShellRenderContext()

  React.useEffect(() => {
    mocks.buddyShellContextSnapshots.push(
      context ? JSON.parse(JSON.stringify(context)) : null
    )
  }, [context])

  return (
    <pre data-testid="buddy-shell-context">
      {JSON.stringify(context)}
    </pre>
  )
}

const render = (ui: React.ReactNode) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  const wrappedUi = (child: React.ReactNode) => (
    <QueryClientProvider client={queryClient}>
      <BuddyShellRenderContextProvider>
        {child}
        <BuddyShellContextProbe />
      </BuddyShellRenderContextProvider>
    </QueryClientProvider>
  )

  const view = rtlRender(wrappedUi(ui))
  return {
    ...view,
    rerender: (nextUi: React.ReactNode) => view.rerender(wrappedUi(nextUi))
  }
}

const readBuddyShellContext = () =>
  JSON.parse(
    screen.getByTestId("buddy-shell-context").textContent || "null"
  )

const openPersonaTab = async (name: string) => {
  fireEvent.click(screen.getByRole("tab", { name }))
}

const waitForLiveSessionPanel = async () => {
  await screen.findByTestId("persona-memory-toggle")
  await screen.findByTestId("persona-resume-session-select")
}

const openStateDocsTab = async () => {
  await openPersonaTab("State Docs")
  await screen.findByTestId("persona-state-editor-toggle-button")
}

const waitForStateDocsEditor = async () => {
  await screen.findByTestId("persona-state-editor-toggle-button")
  if (!screen.queryByTestId("persona-state-soul-input")) {
    fireEvent.click(screen.getByTestId("persona-state-editor-toggle-button"))
  }
  return (await screen.findByTestId("persona-state-soul-input")) as HTMLTextAreaElement
}

class MockWebSocket {
  static instances: MockWebSocket[] = []

  url: string
  binaryType = "blob"
  onopen: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  send = vi.fn<(payload: string) => void>()
  close = vi.fn<() => void>()

  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
    this.close.mockImplementation(() => {
      this.onclose?.({} as CloseEvent)
    })
  }

  emitOpen() {
    this.onopen?.(new Event("open"))
  }

  emitMessage(data: string | ArrayBuffer) {
    this.onmessage?.({ data } as MessageEvent)
  }
}

const getSentPayloads = (ws: MockWebSocket) =>
  ws.send.mock.calls.map(([payload]) => JSON.parse(String(payload)))

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
    } as any)
    mocks.buildPersonaWebSocketUrl.mockReturnValue(
      "ws://persona.test/api/v1/persona/stream"
    )
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

  it("shows connect empty state while offline and navigates to settings", () => {
    mocks.isOnline = false
    render(<SidepanelPersona />)

    expect(screen.getByTestId("sidepanel-header")).toHaveTextContent("Persona Garden")
    expect(screen.getByText("Connect to use Persona")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Settings" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/settings")
  })

  it("shows auth guidance instead of the generic offline copy when credentials are missing", () => {
    mocks.isOnline = false
    mocks.uxState = "error_auth"

    render(<SidepanelPersona />)

    expect(
      screen.getByText("Add your credentials to use Persona")
    ).toBeInTheDocument()
    expect(
      screen.queryByText("Connect to use Persona")
    ).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Settings" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/settings")
  })

  it("shows setup guidance when first-run onboarding is incomplete", () => {
    mocks.isOnline = false
    mocks.uxState = "unconfigured"
    mocks.hasCompletedFirstRun = false

    render(<SidepanelPersona />)

    expect(
      screen.getByText("Finish setup to use Persona")
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Settings" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/settings")
  })

  it("shows an unreachable-server state instead of the generic offline copy", () => {
    mocks.isOnline = false
    mocks.uxState = "error_unreachable"

    render(<SidepanelPersona />)

    expect(
      screen.getByText("Can't reach your tldw server right now")
    ).toBeInTheDocument()
    expect(
      screen.queryByText("Connect to use Persona")
    ).not.toBeInTheDocument()
  })

  it("shows unavailable state when persona capability is missing", () => {
    mocks.capabilitiesState.capabilities = {
      hasPersona: false,
      hasPersonalization: true
    }
    render(<SidepanelPersona />)

    expect(screen.getByTestId("sidepanel-header")).toHaveTextContent("Persona Garden")
    expect(screen.getByText("Persona unavailable")).toBeInTheDocument()
  })

  it("renders Persona Garden framing while keeping live session controls", async () => {
    render(<SidepanelPersona />)

    expect(screen.getByTestId("sidepanel-header")).toHaveTextContent("Persona Garden")
    expect(screen.getByRole("tab", { name: "Live Session" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Profiles" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Voice & Examples" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Visuals" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "State Docs" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Scopes" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Policies" })).toBeInTheDocument()
    await waitForLiveSessionPanel()
  })

  it("renders degraded Persona/Buddy diagnostics from visual runtime state while controls stay available", async () => {
    mocks.location.search = "?persona_id=research_assistant&tab=live"
    mocks.capabilitiesState.capabilities = {
      hasPersona: true,
      hasPersonalization: true,
      hasAudio: true,
      hasMcp: true
    } as any
    usePersonaVisualRuntimeStore.setState({
      runtimeDiagnostics: {
        personaId: "research_assistant",
        sessionId: null,
        packId: "pack-1",
        packTitle: "Broken Pack",
        packLoadStatus: "error",
        diagnostic: {
          code: "load_failed",
          severity: "warning",
          title: "Visual pack did not load",
          message: "The active visual pack could not be loaded."
        },
        updatedAt: 1_765_333_200_000
      }
    } as any)
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            { id: "research_assistant", name: "Research Assistant" }
          ]
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            version: 1,
            buddy_summary: {
              has_buddy: true,
              persona_name: "Research Assistant",
              role_summary: "Keeps research sessions moving",
              visual: null
            },
            voice_defaults: {
              voice_chat_trigger_phrases: ["hey research"],
              wake_behavior: "continuous"
            },
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitForLiveSessionPanel()
    const diagnostics = await screen.findByTestId("persona-buddy-diagnostics")

    expect(diagnostics).toHaveTextContent("Persona Buddy degraded")
    expect(diagnostics).toHaveTextContent("Visual pack did not load")
    expect(diagnostics).toHaveTextContent("Pack ID: pack-1")
    expect(screen.getByRole("button", { name: "Connect" })).toBeInTheDocument()
  })

  it("shows profile load failures in Persona/Buddy diagnostics", async () => {
    mocks.location.search = "?persona_id=research_assistant&tab=live"
    mocks.capabilitiesState.capabilities = {
      hasPersona: true,
      hasPersonalization: true,
      hasMcp: true
    } as any
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            { id: "research_assistant", name: "Research Assistant" }
          ]
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: false,
          error: "Profile service offline",
          json: async () => ({})
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitForLiveSessionPanel()
    const diagnostics = await screen.findByTestId("persona-buddy-diagnostics")

    expect(diagnostics).toHaveTextContent("Persona Buddy degraded")
    expect(diagnostics).toHaveTextContent("Profile")
    expect(diagnostics).toHaveTextContent("Load failed")
    expect(diagnostics).toHaveTextContent("Profile service offline")
    expect(screen.getByRole("button", { name: "Connect" })).toBeInTheDocument()
  })

  it("boots persona selection and active tab from query params", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { body?: any }) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            { id: "research_assistant", name: "Research Assistant" },
            { id: "garden-helper", name: "Garden Helper" }
          ]
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions?persona_id=garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: "sess-garden-helper",
            persona: { id: init?.body?.persona_id || null }
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    expect(screen.getByRole("tab", { name: "Profiles" })).toHaveAttribute(
      "aria-selected",
      "true"
    )

    await openPersonaTab("Live Session")
    fireEvent.click(await screen.findByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(mocks.fetchWithAuth).toHaveBeenCalled()
    })
    const calledPaths = mocks.fetchWithAuth.mock.calls.map(([path]) => String(path))
    expect(
      calledPaths.some((path) => path.includes("/persona/profiles/garden-helper"))
    ).toBe(true)
    expect(
      calledPaths.some((path) =>
        path.includes("/persona/sessions?persona_id=garden-helper")
      )
    ).toBe(true)
    const createSessionCall = mocks.fetchWithAuth.mock.calls.find(
      ([path]) => String(path) === "/api/v1/persona/session"
    )
    expect(createSessionCall?.[1]).toEqual(
      expect.objectContaining({
        method: "POST",
        body: expect.objectContaining({
          persona_id: "garden-helper"
        })
      })
    )
  })

  it("confirms before exporting the selected live session transcript", async () => {
    const createObjectURLSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:persona-session-export")
    const revokeObjectURLSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {})

    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { body?: any }) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions?persona_id=research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path === "/api/v1/persona/session") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: "sess-export",
            persona: { id: init?.body?.persona_id || "research_assistant" }
          })
        })
      }
      if (path === "/api/v1/persona/sessions/sess-export?limit_turns=0") {
        return Promise.resolve({
          ok: true,
          json: async () => ({ preferences: {} })
        })
      }
      if (path === "/api/v1/persona/sessions/sess-export/export") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: "sess-export",
            persona_id: "research_assistant",
            format: "json",
            created_at: "2026-05-21T00:00:00Z",
            updated_at: "2026-05-21T00:00:01Z",
            turn_count: 1,
            redaction_markers: ["metadata.auth_token"],
            turns: [
              {
                turn_id: "turn-1",
                timestamp: "2026-05-21T00:00:01Z",
                role: "user",
                event_type: "user_message",
                content: "hello",
                metadata: {}
              }
            ]
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    try {
      render(<SidepanelPersona />)

      await waitForLiveSessionPanel()
      fireEvent.click(screen.getByRole("button", { name: "Connect" }))

      const exportButton = await screen.findByTestId("persona-transcript-export-button")
      fireEvent.click(exportButton)

      expect(await screen.findByText("Export selected Persona session?")).toBeInTheDocument()
      expect(screen.getByText("Session: sess-export")).toBeInTheDocument()
      expect(
        mocks.fetchWithAuth
      ).not.toHaveBeenCalledWith(
        "/api/v1/persona/sessions/sess-export/export",
        { method: "GET" }
      )
      fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
      expect(createObjectURLSpy).not.toHaveBeenCalled()

      fireEvent.click(exportButton)
      fireEvent.click(await screen.findByRole("button", { name: "Confirm export" }))

      await waitFor(() => {
        expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
          "/api/v1/persona/sessions/sess-export/export",
          { method: "GET" }
        )
      })
      expect(createObjectURLSpy).toHaveBeenCalledTimes(1)
      const blob = createObjectURLSpy.mock.calls[0]?.[0] as Blob
      expect(blob.type).toBe("application/json")
      expect(revokeObjectURLSpy).toHaveBeenCalledWith("blob:persona-session-export")
      expect(
        await screen.findByText("Transcript export downloaded.")
      ).toBeInTheDocument()
    } finally {
      createObjectURLSpy.mockRestore()
      revokeObjectURLSpy.mockRestore()
    }
  })

  it("publishes route-local buddy context ahead of stale persisted assistant state", async () => {
    window.localStorage.setItem(
      SELECTED_ASSISTANT_STORAGE_KEY,
      JSON.stringify({
        kind: "persona",
        id: "stale-persona",
        name: "Stale Persona"
      })
    )
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/setup-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, completed_runs: 0, completion_rate: 0 }
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: 1,
            buddy_summary: {
              has_buddy: true,
              persona_name: "Garden Helper",
              role_summary: "Keeps the route on track",
              visual: {
                species_id: "owl",
                silhouette_id: "perch",
                palette_id: "dawn"
              }
            }
          })
        })
      }
      if (path.includes("/persona/sessions?persona_id=garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(readBuddyShellContext()).toMatchObject({
        surface_id: "persona-garden",
        surface_active: true,
        active_persona_id: "garden-helper",
        position_bucket: "sidepanel-desktop",
        persona_source: "route-local",
        buddy_summary: {
          has_buddy: true,
          persona_name: "Garden Helper",
          role_summary: "Keeps the route on track",
          visual: {
            species_id: "owl",
            silhouette_id: "perch",
            palette_id: "dawn"
          }
        }
      })
    })
  })

  it("clears the previous buddy summary while the next persona profile is still loading", async () => {
    let resolveBuilderProfile: ((value: unknown) => void) | null = null
    const builderProfilePromise = new Promise((resolve) => {
      resolveBuilderProfile = resolve
    })

    mocks.location.search = "?persona_id=garden-helper&tab=profiles"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            { id: "garden-helper", name: "Garden Helper" },
            { id: "builder-bot", name: "Builder Bot" }
          ]
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/setup-analytics") ||
        path.includes("/persona/profiles/builder-bot/setup-analytics")
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: path.includes("builder-bot") ? "builder-bot" : "garden-helper",
            summary: { total_runs: 0, completed_runs: 0, completion_rate: 0 }
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: 1,
            buddy_summary: {
              has_buddy: true,
              persona_name: "Garden Helper",
              role_summary: "Keeps the route on track",
              visual: {
                species_id: "owl",
                silhouette_id: "perch",
                palette_id: "dawn"
              }
            }
          })
        })
      }
      if (path.includes("/persona/profiles/builder-bot")) {
        return builderProfilePromise as Promise<any>
      }
      if (
        path.includes("/persona/sessions?persona_id=garden-helper") ||
        path.includes("/persona/sessions?persona_id=builder-bot")
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    const view = render(<SidepanelPersona />)

    await waitFor(() => {
      expect(readBuddyShellContext()).toMatchObject({
        active_persona_id: "garden-helper",
        buddy_summary: {
          persona_name: "Garden Helper"
        }
      })
    })

    mocks.location.search = "?persona_id=builder-bot&tab=profiles"
    view.rerender(<SidepanelPersona />)

    await waitFor(() => {
      expect(readBuddyShellContext()).toMatchObject({
        active_persona_id: "builder-bot"
      })
      expect(readBuddyShellContext().buddy_summary).toBeNull()
    })

    expect(
      mocks.buddyShellContextSnapshots.some(
        (snapshot) =>
          snapshot?.active_persona_id === "builder-bot" &&
          snapshot?.buddy_summary?.persona_name === "Garden Helper"
      )
    ).toBe(false)

    resolveBuilderProfile?.({
      ok: true,
      json: async () => ({
        id: "builder-bot",
        version: 2,
        buddy_summary: {
          has_buddy: true,
          persona_name: "Builder Bot",
          role_summary: "Keeps builds moving",
          visual: {
            species_id: "robot",
            silhouette_id: "boxy",
            palette_id: "steel"
          }
        }
      })
    })

    await waitFor(() => {
      expect(readBuddyShellContext()).toMatchObject({
        active_persona_id: "builder-bot",
        buddy_summary: {
          persona_name: "Builder Bot"
        }
      })
    })
  })

  it("clears buddy shell context when the persona route drops offline", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/setup-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, completed_runs: 0, completion_rate: 0 }
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: 1,
            buddy_summary: {
              has_buddy: true,
              persona_name: "Garden Helper",
              role_summary: "Keeps the route on track",
              visual: {
                species_id: "owl",
                silhouette_id: "perch",
                palette_id: "dawn"
              }
            }
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    const view = render(<SidepanelPersona />)

    await waitFor(() => {
      expect(readBuddyShellContext()).toMatchObject({
        active_persona_id: "garden-helper",
        buddy_summary: {
          persona_name: "Garden Helper"
        }
      })
    })

    mocks.isOnline = false
    view.rerender(<SidepanelPersona />)

    await waitFor(() => {
      expect(readBuddyShellContext()).toBeNull()
    })
  })

  it("clears buddy shell context when the persona route enters a non-interactive UX state", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/setup-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, completed_runs: 0, completion_rate: 0 }
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: 1,
            buddy_summary: {
              has_buddy: true,
              persona_name: "Garden Helper",
              role_summary: "Keeps the route on track",
              visual: {
                species_id: "owl",
                silhouette_id: "perch",
                palette_id: "dawn"
              }
            }
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    const view = render(<SidepanelPersona />)

    await waitFor(() => {
      expect(readBuddyShellContext()).toMatchObject({
        active_persona_id: "garden-helper"
      })
    })

    mocks.uxState = "error_auth"
    view.rerender(<SidepanelPersona />)

    await waitFor(() => {
      expect(readBuddyShellContext()).toBeNull()
    })
  })

  it("loads and renders setup analytics in profiles", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })

    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/setup-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: {
              total_runs: 4,
              completed_runs: 3,
              completion_rate: 0.75,
              dry_run_completion_count: 2,
              live_session_completion_count: 1,
              most_common_dropoff_step: "commands",
              handoff_click_rate: 0.5,
              handoff_target_reach_rate: 0.25,
              first_post_setup_action_rate: 0.5,
              handoff_target_reached_counts: {},
              detour_started_counts: {},
              detour_returned_counts: {}
            },
            recent_runs: []
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_events: 0, direct_command_count: 0, planner_fallback_count: 0 }
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            voice_defaults: {},
            setup: {
              status: "completed",
              version: 1,
              current_step: "test",
              completed_steps: ["persona", "voice", "commands", "safety", "test"],
              completed_at: "2026-03-14T10:00:00Z",
              last_test_type: "dry_run"
            },
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("persona-setup-analytics-card")).toBeInTheDocument()
    })

    expect(screen.getByTestId("persona-setup-analytics-card")).toHaveTextContent("75%")
    expect(
      mocks.fetchWithAuth.mock.calls.filter(([path]) =>
        String(path).includes("/persona/profiles/garden-helper/setup-analytics")
      )
    ).toHaveLength(1)
  })

  it("hides setup analytics in profiles when the persona has no recorded runs", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })

    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/setup-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: {
              total_runs: 0,
              completed_runs: 0,
              completion_rate: 0,
              dry_run_completion_count: 0,
              live_session_completion_count: 0,
              most_common_dropoff_step: null,
              handoff_click_rate: 0,
              handoff_target_reach_rate: 0,
              first_post_setup_action_rate: 0,
              handoff_target_reached_counts: {},
              detour_started_counts: {},
              detour_returned_counts: {}
            },
            recent_runs: []
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_events: 0, direct_command_count: 0, planner_fallback_count: 0 }
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            voice_defaults: {},
            setup: {
              status: "completed",
              version: 1,
              current_step: "test",
              completed_steps: ["persona", "voice", "commands", "safety", "test"],
              completed_at: "2026-03-14T10:00:00Z",
              last_test_type: "dry_run"
            },
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByText("Assistant Defaults")).toBeInTheDocument()
    })

    expect(screen.queryByTestId("persona-setup-analytics-card")).not.toBeInTheDocument()
    expect(
      mocks.fetchWithAuth.mock.calls.some(([path]) =>
        String(path).includes("/persona/profiles/garden-helper/setup-analytics")
      )
    ).toBe(true)
  })

  it("does not fetch setup analytics outside profiles", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=commands"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })

    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_events: 0, direct_command_count: 0, planner_fallback_count: 0 },
            commands: [],
            fallbacks: { total_invocations: 0 }
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            voice_defaults: {},
            setup: {
              status: "completed",
              version: 1,
              current_step: "test",
              completed_steps: ["persona", "voice", "commands", "safety", "test"],
              completed_at: "2026-03-14T10:00:00Z",
              last_test_type: "dry_run"
            },
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Commands" })).toHaveAttribute(
        "aria-selected",
        "true"
      )
    })

    expect(
      mocks.fetchWithAuth.mock.calls.some(([path]) =>
        String(path).includes("/persona/profiles/garden-helper/setup-analytics")
      )
    ).toBe(false)
  })

  it("fails closed and warns when setup analytics loading fails in profiles", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    try {
      mocks.fetchWithAuth.mockImplementation((path: string) => {
        if (path.includes("/persona/catalog")) {
          return Promise.resolve({
            ok: true,
            json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
          })
        }
        if (path.includes("/persona/profiles/garden-helper/setup-analytics")) {
          return Promise.reject(new Error("setup analytics offline"))
        }
        if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              persona_id: "garden-helper",
              summary: { total_events: 0, direct_command_count: 0, planner_fallback_count: 0 }
            })
          })
        }
        if (path.includes("/persona/profiles/garden-helper")) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              voice_defaults: {},
              setup: {
                status: "completed",
                version: 1,
                current_step: "test",
                completed_steps: ["persona", "voice", "commands", "safety", "test"],
                completed_at: "2026-03-14T10:00:00Z",
                last_test_type: "dry_run"
              },
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: false,
          error: `unhandled path: ${path}`,
          json: async () => ({})
        })
      })

      render(<SidepanelPersona />)

      await waitFor(() => {
        expect(screen.getByText("Assistant Defaults")).toBeInTheDocument()
      })

      expect(screen.queryByTestId("persona-setup-analytics-card")).not.toBeInTheDocument()
      expect(warnSpy).toHaveBeenCalledWith(
        "tldw_server: failed to load persona setup analytics",
        expect.objectContaining({
          personaId: "garden-helper",
          error: expect.any(Error)
        })
      )
    } finally {
      warnSpy.mockRestore()
    }
  })

  it("detours setup into visual buddy setup and returns to the wizard", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })

    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path === "/api/v1/persona/catalog") {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path === "/api/v1/persona/profiles/garden-helper") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: 2,
            voice_defaults: {
              confirmation_mode: "destructive_only"
            },
            setup: {
              status: "in_progress",
              version: 1,
              current_step: "voice",
              completed_steps: ["persona"],
              completed_at: null,
              last_test_type: null
            },
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper/setup-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, completed_runs: 0, completion_rate: 0 }
          })
        })
      }
      if (path === "/api/v1/persona/profiles/garden-helper/visual-packs") {
        return Promise.resolve({
          ok: true,
          json: async () => ({ packs: [], active_pack: null })
        })
      }
      if (path === "/api/v1/persona/visual-starter-packs") {
        return Promise.resolve({
          ok: true,
          json: async () => ({ starter_packs: [] })
        })
      }
      if (path.includes("/visual-library")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ items: [] })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-overlay")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Open Buddy builder" }))

    await waitFor(() => {
      expect(screen.queryByTestId("assistant-setup-overlay")).not.toBeInTheDocument()
    })
    expect(await screen.findByTestId("persona-visual-pack-editor")).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Visuals" })).toBeInTheDocument()
    expect(screen.queryByRole("tab", { name: "Profiles" })).not.toBeInTheDocument()
    expect(screen.queryByRole("tab", { name: "Commands" })).not.toBeInTheDocument()
    expect(screen.queryByRole("tab", { name: "Live Session" })).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Return to setup" }))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-overlay")).toBeInTheDocument()
    })
  })

  it("captures starter command and safety choices into the setup handoff summary", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })

    let profileVersion = 3
    let currentVoiceDefaults = {
      confirmation_mode: "destructive_only"
    }
    let currentSetup = {
      status: "in_progress",
      version: 1,
      current_step: "commands",
      completed_steps: ["persona", "voice"],
      completed_at: null,
      last_test_type: null
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, matched_runs: 0, fallback_runs: 0 }
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands/test") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            heard_text: init?.body?.heard_text,
            matched: true,
            command_name: "Search Notes"
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ id: "cmd-search-notes" })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/connections") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "conn-slack-alerts",
            name: init?.body?.name ?? "Slack Alerts"
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentVoiceDefaults = {
            ...currentVoiceDefaults,
            ...(init?.body?.voice_defaults || {})
          }
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: currentVoiceDefaults,
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: currentVoiceDefaults,
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-overlay")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Search Notes" }))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("safety")
    })

    fireEvent.click(screen.getByRole("button", { name: "Ask for destructive actions" }))
    fireEvent.click(screen.getByRole("button", { name: "Add one connection now" }))
    fireEvent.change(screen.getByLabelText("Connection name"), {
      target: { value: "Slack Alerts" }
    })
    fireEvent.change(screen.getByLabelText("Base URL"), {
      target: { value: "https://hooks.example.com/incoming" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save safety and connection" }))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.change(screen.getByPlaceholderText("Try a spoken phrase"), {
      target: { value: "search notes for project alpha" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run dry-run test" }))

    await screen.findByText(/Matched Search Notes/i)

    fireEvent.click(screen.getByRole("button", { name: "Finish with dry-run test" }))

    await waitFor(() => {
      expect(screen.getByTestId("persona-setup-handoff-card")).toBeInTheDocument()
    })

    expect(screen.getByTestId("persona-setup-handoff-card")).toHaveTextContent(
      "Added 1 starter command"
    )
    expect(screen.getByTestId("persona-setup-handoff-card")).toHaveTextContent(
      "Ask for destructive actions"
    )
    expect(screen.getByTestId("persona-setup-handoff-card")).toHaveTextContent(
      "Connection added: Slack Alerts"
    )
  })

  it("emits setup analytics for setup completion and handoff clicks", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"

    let profileVersion = 3
    let currentVoiceDefaults = {
      confirmation_mode: "destructive_only"
    }
    let currentSetup = {
      status: "in_progress",
      version: 1,
      run_id: "setup-run-1",
      current_step: "commands",
      completed_steps: ["persona", "voice"],
      completed_at: null,
      last_test_type: null
    }
    const setupEventBodies: Array<Record<string, unknown>> = []

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/setup-events") && method === "POST") {
        setupEventBodies.push(init?.body || {})
        return Promise.resolve({
          ok: true,
          json: async () => ({
            event_id: init?.body?.event_id || "evt-1",
            run_id: init?.body?.run_id || "setup-run-1",
            event_type: init?.body?.event_type || "step_viewed",
            deduped: false,
            created_at: "2026-03-14T10:00:00.000Z"
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands/test") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            heard_text: init?.body?.heard_text,
            matched: true,
            command_name: "Search Notes"
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ id: "cmd-search-notes" })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/connections") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "conn-slack-alerts",
            name: init?.body?.name ?? "Slack Alerts"
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentVoiceDefaults = {
            ...currentVoiceDefaults,
            ...(init?.body?.voice_defaults || {})
          }
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: currentVoiceDefaults,
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: currentVoiceDefaults,
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("commands")
    })

    fireEvent.click(screen.getByRole("button", { name: "Search Notes" }))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("safety")
    })

    fireEvent.click(screen.getByRole("button", { name: "Ask for destructive actions" }))
    fireEvent.click(screen.getByRole("button", { name: "Add one connection now" }))
    fireEvent.change(screen.getByLabelText("Connection name"), {
      target: { value: "Slack Alerts" }
    })
    fireEvent.change(screen.getByLabelText("Base URL"), {
      target: { value: "https://hooks.example.com/incoming" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save safety and connection" }))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.change(screen.getByPlaceholderText("Try a spoken phrase"), {
      target: { value: "search notes for project alpha" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run dry-run test" }))

    await screen.findByText(/Matched Search Notes/i)

    fireEvent.click(screen.getByRole("button", { name: "Finish with dry-run test" }))

    await waitFor(() => {
      expect(screen.getByTestId("persona-setup-handoff-card")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Review commands" }))

    await waitFor(() => {
      expect(
        setupEventBodies.some((body) => body.event_type === "setup_completed")
      ).toBe(true)
      expect(
        setupEventBodies.some(
          (body) =>
            body.event_type === "handoff_action_clicked" &&
            body.action_target === "commands"
        )
      ).toBe(true)
    })
  })

  it("smoke-tests setup from persona choice through starter retry and dry-run handoff", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })

    let profileVersion = 2
    let starterAttempts = 0
    let currentVoiceDefaults = {
      confirmation_mode: "destructive_only"
    }
    let currentSetup = {
      status: "in_progress",
      version: 1,
      run_id: "setup-smoke-run",
      current_step: "persona",
      completed_steps: [],
      completed_at: null,
      last_test_type: null
    }
    const setupEventBodies: Array<Record<string, unknown>> = []

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/setup-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: {
              total_runs: 0,
              completion_rate: 0,
              first_post_setup_action_rate: 0
            }
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper/setup-events") && method === "POST") {
        setupEventBodies.push(init?.body || {})
        return Promise.resolve({
          ok: true,
          json: async () => ({
            event_id: init?.body?.event_id || "evt-setup-smoke",
            run_id: init?.body?.run_id || "setup-smoke-run",
            event_type: init?.body?.event_type || "step_viewed",
            deduped: false,
            created_at: "2026-03-14T10:00:00.000Z"
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, matched_runs: 0, fallback_runs: 0 }
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands/test") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            heard_text: init?.body?.heard_text,
            matched: true,
            command_name: "Search Notes"
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "POST"
      ) {
        starterAttempts += 1
        if (starterAttempts === 1) {
          return Promise.resolve({
            ok: false,
            error: "Failed to create starter command",
            json: async () => ({})
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({ id: "cmd-search-notes" })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentVoiceDefaults = {
            ...currentVoiceDefaults,
            ...(init?.body?.voice_defaults || {})
          }
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: currentVoiceDefaults,
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("persona")
    })

    fireEvent.click(await screen.findByRole("button", { name: "Use Garden Helper persona" }))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("voice")
    })

    fireEvent.click(screen.getByRole("button", { name: "Save assistant defaults" }))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("commands")
    })

    fireEvent.click(screen.getByRole("button", { name: "Search Notes" }))

    expect(await screen.findByText("Failed to create starter command")).toBeInTheDocument()
    expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("commands")

    fireEvent.click(screen.getByRole("button", { name: "Retry Search Notes" }))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("safety")
    })

    fireEvent.click(screen.getByRole("button", { name: "Ask for destructive actions" }))
    fireEvent.click(screen.getByRole("button", { name: "No external connections for now" }))
    fireEvent.click(screen.getByRole("button", { name: "Save safety choices" }))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.change(screen.getByPlaceholderText("Try a spoken phrase"), {
      target: { value: "search notes for project alpha" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run dry-run test" }))

    await screen.findByText(/Matched Search Notes/i)

    fireEvent.click(screen.getByRole("button", { name: "Finish with dry-run test" }))

    await waitFor(() => {
      expect(screen.queryByTestId("assistant-setup-overlay")).not.toBeInTheDocument()
    })

    expect(starterAttempts).toBe(2)
    expect(currentSetup.status).toBe("completed")
    expect(currentSetup.completed_steps).toEqual([
      "persona",
      "voice",
      "commands",
      "safety",
      "test"
    ])
    expect(screen.getByTestId("persona-setup-handoff-card")).toHaveTextContent(
      "Assistant setup complete"
    )
    expect(screen.getByTestId("persona-setup-handoff-card")).toHaveTextContent(
      "Added 1 starter command"
    )
    expect(screen.getByTestId("persona-setup-handoff-card")).toHaveTextContent(
      "Ask for destructive actions"
    )
    await waitFor(() => {
      expect(
        setupEventBodies.some(
          (body) =>
            body.event_type === "setup_completed" &&
            body.completion_type === "dry_run"
        )
      ).toBe(true)
    })
  })

  it("keeps the commands step in place with retry guidance when starter creation fails", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=commands"

    let profileVersion = 2
    let currentSetup = {
      status: "in_progress",
      current_step: "commands",
      completed_steps: ["persona", "voice"]
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: false,
          error: "Failed to create starter command",
          json: async () => ({})
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: {
                confirmation_mode: "destructive_only"
              },
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: {
              confirmation_mode: "destructive_only"
            },
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("commands")
    })

    fireEvent.click(screen.getByRole("button", { name: "Search Notes" }))

    expect(await screen.findByText("Failed to create starter command")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Try a starter template again, add an MCP starter instead, or continue without starter commands."
      )
    ).toBeInTheDocument()
    expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("commands")

    fireEvent.click(screen.getByRole("button", { name: "Continue without starter commands" }))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("safety")
    })
  })

  it("keeps the safety step in place with retry guidance when setup connection creation fails", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"

    let profileVersion = 2
    let currentSetup = {
      status: "in_progress",
      current_step: "safety",
      completed_steps: ["persona", "voice", "commands"]
    }
    let currentVoiceDefaults = {
      confirmation_mode: "destructive_only"
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      if (
        path.includes("/persona/profiles/garden-helper/connections") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: false,
          error: "Failed to create setup connection",
          json: async () => ({})
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentVoiceDefaults = {
            ...currentVoiceDefaults,
            ...(init?.body?.voice_defaults || {})
          }
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: currentVoiceDefaults,
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: currentVoiceDefaults,
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("safety")
    })

    fireEvent.click(screen.getByRole("button", { name: "Ask for destructive actions" }))
    fireEvent.click(screen.getByRole("button", { name: "Add one connection now" }))
    fireEvent.change(screen.getByLabelText("Connection name"), {
      target: { value: "Slack Alerts" }
    })
    fireEvent.change(screen.getByLabelText("Base URL"), {
      target: { value: "https://hooks.example.com/incoming" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save safety and connection" }))

    expect(await screen.findByText("Failed to create setup connection")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Fix the connection details below and try again, or skip external connections for now."
      )
    ).toBeInTheDocument()
    expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("safety")

    fireEvent.click(screen.getByRole("button", { name: "No external connections for now" }))
    fireEvent.click(screen.getByRole("button", { name: "Save safety choices" }))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })
  })

  it("renders a local live-send failure outcome during the setup test step", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=live"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/sessions?")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path === "/api/v1/persona/session") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: "sess-setup-live",
            persona: { id: "garden-helper" }
          })
        })
      }
      if (path.includes("/persona/sessions/sess-setup-live")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ preferences: {} })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: 2,
            voice_defaults: {
              confirmation_mode: "destructive_only"
            },
            setup: {
              status: "in_progress",
              current_step: "test",
              completed_steps: ["persona", "voice", "commands", "safety"]
            },
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.click(screen.getByRole("button", { name: "Connect live session" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    await screen.findByPlaceholderText("Try a live message")

    ws.send.mockImplementation((payload: string) => {
      const parsed = JSON.parse(String(payload))
      if (parsed.type === "user_message") {
        throw new Error("Socket send failed")
      }
    })

    fireEvent.change(screen.getByPlaceholderText("Try a live message"), {
      target: { value: "summarize my assistant setup" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send live test" }))

    expect(await screen.findByText("Socket send failed")).toBeInTheDocument()
    expect(
      screen.getByText("Try sending the live test again or reconnect the live session.")
    ).toBeInTheDocument()
  })

  it("surfaces live_unavailable on setup connect failure and autoconnects on the detour", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=live"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })

    let connectAttempts = 0

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/sessions?")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path === "/api/v1/persona/session") {
        connectAttempts += 1
        if (connectAttempts === 1) {
          return Promise.resolve({
            ok: false,
            error: "Failed to create persona session",
            json: async () => ({})
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: "sess-setup-live",
            persona: { id: "garden-helper" }
          })
        })
      }
      if (path.includes("/persona/sessions/sess-setup-live")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ preferences: {} })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: 2,
            voice_defaults: {
              confirmation_mode: "destructive_only"
            },
            setup: {
              status: "in_progress",
              current_step: "test",
              completed_steps: ["persona", "voice", "commands", "safety"]
            },
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.click(screen.getByRole("button", { name: "Connect live session" }))

    expect(
      await screen.findByText(/Live session unavailable until you connect/i)
    ).toBeInTheDocument()
    expect(MockWebSocket.instances).toHaveLength(0)

    fireEvent.click(screen.getByRole("button", { name: "Open Live Session to fix this" }))

    await waitFor(() => {
      expect(screen.queryByTestId("assistant-setup-overlay")).not.toBeInTheDocument()
    })
    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    expect(
      screen.getByText("Finish this live test, then return to setup.")
    ).toBeInTheDocument()
  })

  it("detours setup into live for a setup live failure and returns manually", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=live"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/sessions?")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path === "/api/v1/persona/session") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: "sess-setup-live",
            persona: { id: "garden-helper" }
          })
        })
      }
      if (path.includes("/persona/sessions/sess-setup-live")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ preferences: {} })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: 2,
            voice_defaults: {
              confirmation_mode: "destructive_only"
            },
            setup: {
              status: "in_progress",
              current_step: "test",
              completed_steps: ["persona", "voice", "commands", "safety"]
            },
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.click(screen.getByRole("button", { name: "Connect live session" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    await screen.findByPlaceholderText("Try a live message")

    ws.send.mockImplementation((payload: string) => {
      const parsed = JSON.parse(String(payload))
      if (parsed.type === "user_message") {
        throw new Error("Socket send failed")
      }
    })

    fireEvent.change(screen.getByPlaceholderText("Try a live message"), {
      target: { value: "summarize my assistant setup" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send live test" }))

    expect(await screen.findByText("Socket send failed")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Try again in Live Session" }))

    await waitFor(() => {
      expect(screen.queryByTestId("assistant-setup-overlay")).not.toBeInTheDocument()
    })
    expect(MockWebSocket.instances).toHaveLength(1)
    expect(
      screen.getByText("Finish this live test, then return to setup.")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Return to setup" }))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })
    expect(
      screen.getByText("Live session is still available if you want to retry.")
    ).toBeInTheDocument()
  })

  it("auto-returns setup from live detour after a successful live response", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=live"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/sessions?")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path === "/api/v1/persona/session") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: "sess-setup-live",
            persona: { id: "garden-helper" }
          })
        })
      }
      if (path.includes("/persona/sessions/sess-setup-live")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ preferences: {} })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: 2,
            voice_defaults: {
              confirmation_mode: "destructive_only"
            },
            setup: {
              status: "in_progress",
              current_step: "test",
              completed_steps: ["persona", "voice", "commands", "safety"]
            },
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.click(screen.getByRole("button", { name: "Connect live session" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    await screen.findByPlaceholderText("Try a live message")

    let setupSendAttempts = 0
    ws.send.mockImplementation((payload: string) => {
      const parsed = JSON.parse(String(payload))
      if (parsed.type === "user_message") {
        setupSendAttempts += 1
        if (setupSendAttempts === 1) {
          throw new Error("Socket send failed")
        }
      }
    })

    fireEvent.change(screen.getByPlaceholderText("Try a live message"), {
      target: { value: "summarize my assistant setup" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send live test" }))

    expect(await screen.findByText("Socket send failed")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Try again in Live Session" }))

    await waitFor(() => {
      expect(screen.queryByTestId("assistant-setup-overlay")).not.toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("Ask Persona..."), {
      target: { value: "summarize my assistant setup" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))

    ws.emitMessage(
      JSON.stringify({
        event: "assistant_delta",
        text_delta: "Here is the answer from the live session."
      })
    )

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })
    expect(
      screen.getByText("Live session responded. Finish setup when you're ready.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Finish with live session" })).toBeInTheDocument()
  })

  it("clears the setup live detour when setup is reset", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=live"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })

    let profileVersion = 2
    let currentSetup = {
      status: "in_progress",
      version: 1,
      current_step: "test",
      completed_steps: ["persona", "voice", "commands", "safety"],
      completed_at: null,
      last_test_type: null
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/sessions?")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path === "/api/v1/persona/session") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: "sess-setup-live",
            persona: { id: "garden-helper" }
          })
        })
      }
      if (path.includes("/persona/sessions/sess-setup-live")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ preferences: {} })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: {
                confirmation_mode: "destructive_only"
              },
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: {
              confirmation_mode: "destructive_only"
            },
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.click(screen.getByRole("button", { name: "Connect live session" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    await screen.findByPlaceholderText("Try a live message")

    ws.send.mockImplementation((payload: string) => {
      const parsed = JSON.parse(String(payload))
      if (parsed.type === "user_message") {
        throw new Error("Socket send failed")
      }
    })

    fireEvent.change(screen.getByPlaceholderText("Try a live message"), {
      target: { value: "summarize my assistant setup" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send live test" }))

    expect(await screen.findByText("Socket send failed")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Try again in Live Session" }))

    await waitFor(() => {
      expect(screen.queryByTestId("assistant-setup-overlay")).not.toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("tab", { name: "Profiles" }))
    expect(screen.getByRole("button", { name: "Reset setup" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Reset setup" }))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("persona")
    })
    expect(
      screen.queryByText("Finish this live test, then return to setup.")
    ).not.toBeInTheDocument()
  })

  it("detours setup into commands for a dry-run no-match and returns to test after save", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=live"

    let profileVersion = 2
    let currentSetup = {
      status: "in_progress",
      version: 1,
      current_step: "test",
      completed_steps: ["persona", "voice", "commands", "safety"],
      completed_at: null,
      last_test_type: null
    }
    const currentVoiceDefaults = {
      confirmation_mode: "destructive_only"
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, matched_runs: 0, fallback_runs: 0 }
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands/test") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            heard_text: init?.body?.heard_text,
            matched: false,
            failure_phase: "planner_fallback"
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            commands: []
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/connections") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "cmd-created-from-setup",
            persona_id: "garden-helper",
            name: init?.body?.name,
            phrases: init?.body?.phrases,
            action_type: init?.body?.action_type,
            action_config: init?.body?.action_config,
            priority: init?.body?.priority,
            enabled: init?.body?.enabled,
            requires_confirmation: init?.body?.requires_confirmation
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: currentVoiceDefaults,
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: currentVoiceDefaults,
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.change(screen.getByPlaceholderText("Try a spoken phrase"), {
      target: { value: "open the pod bay doors" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run dry-run test" }))

    await screen.findByText(/No direct command matched/i)

    fireEvent.click(screen.getByRole("button", { name: "Create command from this phrase" }))

    await waitFor(() => {
      expect(screen.queryByTestId("assistant-setup-overlay")).not.toBeInTheDocument()
    })
    expect(screen.getByTestId("persona-commands-draft-banner")).toHaveTextContent(
      "Drafted from assistant setup"
    )
    expect(screen.getByTestId("persona-commands-name-input")).toHaveValue(
      "Open the pod bay doors"
    )

    fireEvent.change(screen.getByTestId("persona-commands-name-input"), {
      target: { value: "Open Pod Bay Doors" }
    })
    fireEvent.change(screen.getByTestId("persona-commands-action-type-select"), {
      target: { value: "custom" }
    })
    fireEvent.change(screen.getByTestId("persona-commands-custom-action-input"), {
      target: { value: "open_pod_bay_doors" }
    })
    fireEvent.click(screen.getByTestId("persona-commands-save"))

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })
    expect(screen.getByPlaceholderText("Try a spoken phrase")).toHaveValue(
      "open the pod bay doors"
    )
    expect(
      screen.getByText("Command saved. Run the same phrase again to confirm setup.")
    ).toBeInTheDocument()
  })

  it("derives handoff review details when setup resumes on the test step", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })

    let profileVersion = 2
    let currentSetup = {
      status: "in_progress",
      version: 1,
      current_step: "test",
      completed_steps: ["persona", "voice", "commands", "safety"],
      completed_at: null,
      last_test_type: null
    }
    const currentVoiceDefaults = {
      confirmation_mode: "never"
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, matched_runs: 0, fallback_runs: 0 }
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands/test") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            heard_text: init?.body?.heard_text,
            matched: true,
            command_name: "Search Notes"
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            commands: [
              {
                id: "cmd-search-notes",
                persona_id: "garden-helper",
                name: "Search Notes",
                phrases: ["search notes for {topic}"],
                action_type: "mcp_tool",
                action_config: { tool_name: "notes.search" },
                priority: 50,
                enabled: true,
                requires_confirmation: false,
                description: "Find notes related to a spoken topic"
              }
            ]
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/connections") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            {
              id: "conn-slack-alerts",
              persona_id: "garden-helper",
              name: "Slack Alerts",
              base_url: "https://hooks.example.com/incoming",
              auth_type: "bearer"
            }
          ]
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: currentVoiceDefaults,
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: currentVoiceDefaults,
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.change(screen.getByPlaceholderText("Try a spoken phrase"), {
      target: { value: "search notes for project alpha" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run dry-run test" }))

    await screen.findByText(/Matched Search Notes/i)

    fireEvent.click(screen.getByRole("button", { name: "Finish with dry-run test" }))

    await waitFor(() => {
      expect(screen.getByTestId("persona-setup-handoff-card")).toBeInTheDocument()
    })

    expect(screen.getByTestId("persona-setup-handoff-card")).toHaveTextContent(
      "1 command available"
    )
    expect(screen.getByTestId("persona-setup-handoff-card")).toHaveTextContent(
      "Never ask"
    )
    expect(screen.getByTestId("persona-setup-handoff-card")).toHaveTextContent(
      "Connection available: Slack Alerts"
    )
  })

  it("renders the setup handoff card when setup returns to the connections tab", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=connections"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })

    let profileVersion = 2
    let currentSetup = {
      status: "in_progress",
      version: 1,
      current_step: "test",
      completed_steps: ["persona", "voice", "commands", "safety"],
      completed_at: null,
      last_test_type: null
    }
    const currentVoiceDefaults = {
      confirmation_mode: "destructive_only"
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, matched_runs: 0, fallback_runs: 0 }
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands/test") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            heard_text: init?.body?.heard_text,
            matched: true,
            command_name: "Search Notes"
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ commands: [] })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/connections") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: currentVoiceDefaults,
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: currentVoiceDefaults,
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.change(screen.getByPlaceholderText("Try a spoken phrase"), {
      target: { value: "search notes for project alpha" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run dry-run test" }))

    await screen.findByText(/Matched Search Notes/i)

    fireEvent.click(screen.getByRole("button", { name: "Finish with dry-run test" }))

    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Connections" })).toHaveAttribute(
        "aria-selected",
        "true"
      )
    })

    expect(screen.getByTestId("persona-setup-handoff-card")).toBeInTheDocument()
  })

  it("keeps the setup handoff visible and targets confirmation mode after a same-tab handoff action", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"

    let profileVersion = 2
    let currentSetup = {
      status: "in_progress",
      version: 1,
      current_step: "test",
      completed_steps: ["persona", "voice", "commands", "safety"],
      completed_at: null,
      last_test_type: null
    }
    const currentVoiceDefaults = {
      confirmation_mode: "destructive_only"
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, matched_runs: 0, fallback_runs: 0 }
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands/test") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            heard_text: init?.body?.heard_text,
            matched: true,
            command_name: "Search Notes"
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ commands: [] })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/connections") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: currentVoiceDefaults,
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: currentVoiceDefaults,
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.change(screen.getByPlaceholderText("Try a spoken phrase"), {
      target: { value: "search notes for project alpha" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run dry-run test" }))

    await screen.findByText(/Matched Search Notes/i)

    fireEvent.click(screen.getByRole("button", { name: "Finish with dry-run test" }))

    await waitFor(() => {
      expect(screen.getByTestId("persona-setup-handoff-card")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Review safety defaults" }))

    expect(screen.getByRole("tab", { name: "Profiles" })).toHaveAttribute("aria-selected", "true")
    expect(screen.getByTestId("persona-setup-handoff-card")).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByLabelText("Confirmation mode")).toHaveFocus()
    })
  })

  it("retargets the setup handoff and opens the command form for the add-command CTA", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=connections"

    let profileVersion = 2
    let currentSetup = {
      status: "in_progress",
      version: 1,
      current_step: "test",
      completed_steps: ["persona", "voice", "commands", "safety"],
      completed_at: null,
      last_test_type: null
    }
    const currentVoiceDefaults = {
      confirmation_mode: "destructive_only"
    }
    const setupEventBodies: Array<Record<string, unknown>> = []

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/setup-events") && method === "POST") {
        setupEventBodies.push(init?.body || {})
        return Promise.resolve({
          ok: true,
          json: async () => ({
            event_id: init?.body?.event_id || "evt-1",
            run_id: init?.body?.run_id || "setup-run-1",
            event_type: init?.body?.event_type || "step_viewed",
            deduped: false,
            created_at: "2026-03-14T10:00:00.000Z"
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, matched_runs: 0, fallback_runs: 0 }
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands/test") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            heard_text: init?.body?.heard_text,
            matched: true,
            command_name: "Search Notes"
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ commands: [] })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/connections") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: currentVoiceDefaults,
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: currentVoiceDefaults,
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.change(screen.getByPlaceholderText("Try a spoken phrase"), {
      target: { value: "search notes for project alpha" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run dry-run test" }))

    await screen.findByText(/Matched Search Notes/i)

    fireEvent.click(screen.getByRole("button", { name: "Finish with dry-run test" }))

    await waitFor(() => {
      expect(screen.getByTestId("persona-setup-handoff-card")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Open Commands" }))

    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Commands" })).toHaveAttribute(
        "aria-selected",
        "true"
      )
    })
    expect(screen.getByTestId("persona-setup-handoff-card")).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByTestId("persona-commands-name-input")).toHaveFocus()
    })
    await waitFor(() => {
      expect(
        setupEventBodies.filter(
          (body) =>
            body.event_type === "handoff_target_reached" &&
            body.action_target === "commands.command_form"
        )
      ).toHaveLength(1)
    })

    fireEvent.click(screen.getByRole("button", { name: "Review commands" }))

    await waitFor(() => {
      expect(screen.getByTestId("persona-commands-name-input")).toHaveFocus()
    })
    await waitFor(() => {
      expect(
        setupEventBodies.filter(
          (body) =>
            body.event_type === "handoff_target_reached" &&
            body.action_target === "commands.command_list"
        )
      ).toHaveLength(1)
    })
  })

  it("retargets the setup handoff and targets the saved connection action after a cross-tab handoff action", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"

    let profileVersion = 2
    let currentSetup = {
      status: "in_progress",
      version: 1,
      current_step: "test",
      completed_steps: ["persona", "voice", "commands", "safety"],
      completed_at: null,
      last_test_type: null
    }
    const currentVoiceDefaults = {
      confirmation_mode: "destructive_only"
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, matched_runs: 0, fallback_runs: 0 }
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands/test") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            heard_text: init?.body?.heard_text,
            matched: true,
            command_name: "Search Notes"
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ commands: [] })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/connections") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            {
              id: "conn-existing",
              persona_id: "garden-helper",
              name: "Existing API",
              base_url: "https://existing.example.com",
              auth_type: "bearer"
            }
          ]
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: currentVoiceDefaults,
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: currentVoiceDefaults,
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.change(screen.getByPlaceholderText("Try a spoken phrase"), {
      target: { value: "search notes for project alpha" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run dry-run test" }))

    await screen.findByText(/Matched Search Notes/i)

    fireEvent.click(screen.getByRole("button", { name: "Finish with dry-run test" }))

    await waitFor(() => {
      expect(screen.getByTestId("persona-setup-handoff-card")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Open connections" }))

    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Connections" })).toHaveAttribute(
        "aria-selected",
        "true"
      )
    })
    expect(screen.getByTestId("persona-setup-handoff-card")).toBeInTheDocument()
    await waitFor(() => {
      expect(
        screen.getByTestId("persona-connections-edit-conn-existing")
      ).toHaveFocus()
    })
  })

  it("retargets the setup handoff and targets the dry-run form after a cross-tab handoff action", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=profiles"

    let profileVersion = 2
    let currentSetup = {
      status: "in_progress",
      version: 1,
      current_step: "test",
      completed_steps: ["persona", "voice", "commands", "safety"],
      completed_at: null,
      last_test_type: null
    }
    const currentVoiceDefaults = {
      confirmation_mode: "destructive_only"
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, matched_runs: 0, fallback_runs: 0 }
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands/test") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            heard_text: init?.body?.heard_text,
            matched: true,
            command_name: "Search Notes"
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ commands: [] })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/connections") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: currentVoiceDefaults,
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: currentVoiceDefaults,
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.change(screen.getByPlaceholderText("Try a spoken phrase"), {
      target: { value: "search notes for project alpha" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run dry-run test" }))

    await screen.findByText(/Matched Search Notes/i)

    fireEvent.click(screen.getByRole("button", { name: "Finish with dry-run test" }))

    await waitFor(() => {
      expect(screen.getByTestId("persona-setup-handoff-card")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Open Test Lab" }))

    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Test Lab" })).toHaveAttribute(
        "aria-selected",
        "true"
      )
    })
    expect(screen.getByTestId("persona-setup-handoff-card")).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByTestId("persona-test-lab-heard-input")).toHaveFocus()
    })
  })

  it("does not collapse the setup handoff when a post-setup dry-run does not match", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=test-lab"

    let profileVersion = 2
    let currentSetup = {
      status: "in_progress",
      version: 1,
      run_id: "setup-run-1",
      current_step: "test",
      completed_steps: ["persona", "voice", "commands", "safety"],
      completed_at: null,
      last_test_type: null
    }
    const currentVoiceDefaults = {
      confirmation_mode: "destructive_only"
    }
    const setupEventBodies: Array<Record<string, unknown>> = []

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/setup-events") && method === "POST") {
        setupEventBodies.push(init?.body || {})
        return Promise.resolve({
          ok: true,
          json: async () => ({
            event_id: init?.body?.event_id || "evt-1",
            run_id: init?.body?.run_id || "setup-run-1",
            event_type: init?.body?.event_type || "step_viewed",
            deduped: false,
            created_at: "2026-03-14T10:00:00.000Z"
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, matched_runs: 0, fallback_runs: 0 }
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands/test") &&
        method === "POST"
      ) {
        const heardText = String(init?.body?.heard_text || "")
        if (heardText === "start a focused research sprint") {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              heard_text: heardText,
              matched: false,
              failure_phase: "planner_fallback"
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            heard_text: heardText,
            matched: true,
            command_name: "Search Notes"
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: currentVoiceDefaults,
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: currentVoiceDefaults,
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.change(screen.getByPlaceholderText("Try a spoken phrase"), {
      target: { value: "search notes for project alpha" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run dry-run test" }))

    await screen.findByText(/Matched Search Notes/i)

    fireEvent.click(screen.getByRole("button", { name: "Finish with dry-run test" }))

    await waitFor(() => {
      expect(screen.getByTestId("persona-setup-handoff-card")).toHaveTextContent(
        "Assistant setup complete"
      )
    })

    fireEvent.change(screen.getByTestId("persona-test-lab-heard-input"), {
      target: { value: "start a focused research sprint" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run dry-run" }))

    await screen.findByText("persona fallback")
    expect(screen.getByTestId("persona-setup-handoff-card")).toHaveTextContent(
      "Assistant setup complete"
    )
    expect(
      setupEventBodies.some(
        (body) =>
          body.event_type === "first_post_setup_action" &&
          body.action_target === "test-lab"
      )
    ).toBe(false)
  })

  it("collapses the setup handoff after the first successful post-setup action", async () => {
    mocks.location.search = "?persona_id=garden-helper&tab=connections"

    let profileVersion = 2
    let currentSetup = {
      status: "in_progress",
      version: 1,
      run_id: "setup-run-1",
      current_step: "test",
      completed_steps: ["persona", "voice", "commands", "safety"],
      completed_at: null,
      last_test_type: null
    }
    const currentVoiceDefaults = {
      confirmation_mode: "destructive_only"
    }
    const setupEventBodies: Array<Record<string, unknown>> = []

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "garden-helper", name: "Garden Helper" }]
        })
      }
      if (path.includes("/persona/profiles/garden-helper/setup-events") && method === "POST") {
        setupEventBodies.push(init?.body || {})
        return Promise.resolve({
          ok: true,
          json: async () => ({
            event_id: init?.body?.event_id || "evt-1",
            run_id: init?.body?.run_id || "setup-run-1",
            event_type: init?.body?.event_type || "step_viewed",
            deduped: false,
            created_at: "2026-03-14T10:00:00.000Z"
          })
        })
      }
      if (path.includes("/persona/profiles/garden-helper/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "garden-helper",
            summary: { total_runs: 0, matched_runs: 0, fallback_runs: 0 }
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands/test") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            heard_text: init?.body?.heard_text,
            matched: true,
            command_name: "Search Notes"
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ commands: [] })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/voice-commands") &&
        method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "cmd-hello",
            persona_id: "garden-helper",
            name: init?.body?.name ?? "Hello there",
            phrases: init?.body?.phrases ?? ["hello there"],
            action_type: init?.body?.action_type ?? "custom",
            action_config: init?.body?.action_config ?? { action: "say_hello" },
            priority: 50,
            enabled: true,
            requires_confirmation: false
          })
        })
      }
      if (
        path.includes("/persona/profiles/garden-helper/connections") &&
        method === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/profiles/garden-helper")) {
        if (method === "PATCH") {
          profileVersion += 1
          currentSetup = {
            ...currentSetup,
            ...(init?.body?.setup || {})
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "garden-helper",
              version: profileVersion,
              voice_defaults: currentVoiceDefaults,
              setup: currentSetup,
              use_persona_state_context_default: true
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "garden-helper",
            version: profileVersion,
            voice_defaults: currentVoiceDefaults,
            setup: currentSetup,
            use_persona_state_context_default: true
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitFor(() => {
      expect(screen.getByTestId("assistant-setup-current-step")).toHaveTextContent("test")
    })

    fireEvent.change(screen.getByPlaceholderText("Try a spoken phrase"), {
      target: { value: "search notes for project alpha" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run dry-run test" }))

    await screen.findByText(/Matched Search Notes/i)

    fireEvent.click(screen.getByRole("button", { name: "Finish with dry-run test" }))

    await waitFor(() => {
      expect(screen.getByTestId("persona-setup-handoff-card")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Review commands" }))

    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Commands" })).toHaveAttribute(
        "aria-selected",
        "true"
      )
    })

    fireEvent.change(screen.getByTestId("persona-commands-name-input"), {
      target: { value: "Hello there" }
    })
    fireEvent.change(screen.getByTestId("persona-commands-phrases-input"), {
      target: { value: "hello there" }
    })
    fireEvent.change(screen.getByTestId("persona-commands-action-type-select"), {
      target: { value: "custom" }
    })
    fireEvent.change(screen.getByTestId("persona-commands-custom-action-input"), {
      target: { value: "say_hello" }
    })
    fireEvent.click(screen.getByTestId("persona-commands-save"))

    await waitFor(() => {
      expect(screen.getByText("Setup complete")).toBeInTheDocument()
      expect(screen.getByText("Add your first command")).toBeInTheDocument()
    })

    expect(
      setupEventBodies.some(
        (body) =>
          body.event_type === "first_post_setup_action" &&
          body.action_target === "commands"
      )
    ).toBe(true)
  })

  it("renders a dedicated companion conversation mode", () => {
    render(<SidepanelPersona mode="companion" />)

    expect(screen.getByTestId("sidepanel-header")).toHaveTextContent("Companion")
    expect(screen.getByPlaceholderText("Ask Companion...")).toBeInTheDocument()
    expect(screen.queryByLabelText("Select persona")).not.toBeInTheDocument()
    expect(screen.queryByTestId("persona-companion-context-toggle")).not.toBeInTheDocument()
    expect(screen.queryByTestId("persona-state-context-toggle")).not.toBeInTheDocument()
    expect(
      screen.queryByTestId("persona-state-editor-toggle-button")
    ).not.toBeInTheDocument()
  })

  it("records the current draft as a companion check-in", async () => {
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { body?: any }) => {
      if (path.includes("/api/v1/companion/check-ins")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "activity-checkin-1",
            event_type: "companion_check_in_recorded",
            source_type: "companion_check_in",
            source_id: "checkin-1",
            surface: String(init?.body?.surface || "companion.workspace"),
            tags: [],
            provenance: {
              capture_mode: "explicit",
              route: "/api/v1/companion/check-ins",
              action: "manual_check_in"
            },
            metadata: {
              summary: String(init?.body?.summary || "")
            },
            created_at: "2026-03-10T12:30:00Z"
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    const draft = "Log this as an explicit companion check-in from persona."
    fireEvent.change(screen.getByPlaceholderText("Ask Persona..."), {
      target: { value: draft }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save check-in" }))

    await waitFor(() => {
      expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
        "/api/v1/companion/check-ins",
        {
          method: "POST",
          body: {
            summary: draft,
            surface: "persona.sidepanel"
          }
        }
      )
    })
    expect(screen.getByText("Saved draft to companion")).toBeInTheDocument()
    expect(screen.getByPlaceholderText("Ask Persona...")).toHaveValue(draft)
  })

  it("records companion-mode drafts with the companion conversation surface", async () => {
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { body?: any }) => {
      if (path.includes("/api/v1/companion/check-ins")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "activity-checkin-2",
            event_type: "companion_check_in_recorded",
            source_type: "companion_check_in",
            source_id: "checkin-2",
            surface: String(init?.body?.surface || "companion.workspace"),
            tags: [],
            provenance: {
              capture_mode: "explicit",
              route: "/api/v1/companion/check-ins",
              action: "manual_check_in"
            },
            metadata: {
              summary: String(init?.body?.summary || "")
            },
            created_at: "2026-03-10T12:45:00Z"
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona mode="companion" />)

    const draft = "Capture this from the dedicated companion conversation."
    fireEvent.change(screen.getByPlaceholderText("Ask Companion..."), {
      target: { value: draft }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save check-in" }))

    await waitFor(() => {
      expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
        "/api/v1/companion/check-ins",
        {
          method: "POST",
          body: {
            summary: draft,
            surface: "companion.conversation"
          }
        }
      )
    })
    expect(screen.getByText("Saved draft to companion")).toBeInTheDocument()
  })

  it("renders companion conversation prompt chips and inserts text into the draft", async () => {
    render(<SidepanelPersona mode="companion" />)

    const chip = await screen.findByRole("button", { name: "Next concrete step" })
    fireEvent.click(chip)

    expect(screen.getByPlaceholderText("Ask Companion...")).toHaveValue(
      "What is the next concrete step for project alpha?"
    )
  })

  it("does not auto-send when a companion prompt chip is clicked", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    })
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { body?: any }) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions?")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path === "/api/v1/persona/session") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: "sess-companion",
            persona: { id: "research_assistant" }
          })
        })
      }
      if (path.includes("/persona/sessions/sess-companion")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ preferences: {} })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona mode="companion" />)

    fireEvent.click(await screen.findByRole("button", { name: "Connect" }))
    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    const ws = MockWebSocket.instances[0]
    ws.emitOpen()
    await screen.findByText("Persona stream connected")

    fireEvent.click(await screen.findByRole("button", { name: "Next concrete step" }))

    expect(getSentPayloads(ws).some((payload) => payload.type === "user_message")).toBe(
      false
    )
    expect(screen.getByPlaceholderText("Ask Companion...")).toHaveValue(
      "What is the next concrete step for project alpha?"
    )
  })

  it("shows a consent-required error when saving a companion check-in without opt-in", async () => {
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/api/v1/companion/check-ins")) {
        return Promise.resolve({
          ok: false,
          status: 409,
          error: "Enable personalization before using companion.",
          json: async () => ({ detail: "Enable personalization before using companion." })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    fireEvent.change(screen.getByPlaceholderText("Ask Persona..."), {
      target: { value: "Do not save this silently." }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save check-in" }))

    expect(
      await screen.findByText("Enable personalization before saving to companion.")
    ).toBeInTheDocument()
  })

  it.each([390, 1280])(
    "keeps new-session and memory controls discoverable at %ipx viewport width",
    (width) => {
      Object.defineProperty(window, "innerWidth", {
        configurable: true,
        writable: true,
        value: width
      })
      window.dispatchEvent(new Event("resize"))

      render(<SidepanelPersona />)

      expect(screen.getByLabelText("Resume session")).toBeInTheDocument()
      expect(screen.getByTestId("persona-memory-toggle")).toBeInTheDocument()
      expect(screen.getByTestId("persona-state-context-toggle")).toBeInTheDocument()
      expect(
        screen.getByTestId("persona-state-context-default-toggle")
      ).toBeInTheDocument()
      expect(screen.getByLabelText("Memory results")).toBeInTheDocument()
      expect(screen.getByText("Memory results: 3")).toBeInTheDocument()
      expect(screen.queryByText("k=3")).not.toBeInTheDocument()
    }
  )

  it("connects persona websocket and supports message/plan confirm-cancel flow", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-12345" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    fireEvent.click(await screen.findByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(mocks.fetchWithAuth).toHaveBeenCalled()
    })
    const calledPaths = mocks.fetchWithAuth.mock.calls.map(([path]) => String(path))
    expect(calledPaths.some((path) => path.includes("/persona/sessions"))).toBe(true)
    expect(calledPaths.some((path) => path.includes("/persona/session"))).toBe(true)
    expect(calledPaths.some((path) => path.includes("/persona/catalog"))).toBe(true)
    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    expect(mocks.buildPersonaWebSocketUrl).toHaveBeenCalledTimes(1)

    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    await screen.findByText("Persona stream connected")

    fireEvent.change(screen.getByPlaceholderText("Ask Persona..."), {
      target: { value: "hello persona" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))

    await waitFor(() => {
      const sentPayloads = getSentPayloads(ws)
      expect(
        sentPayloads.some(
          (payload) =>
            payload.type === "user_message" &&
            payload.session_id === "sess-12345" &&
            payload.text === "hello persona"
        )
      ).toBe(true)
    })

    ws.emitMessage(
      JSON.stringify({
        event: "tool_plan",
        plan_id: "plan-1",
        steps: [
          { idx: 0, tool: "ingest_url", description: "ingest" },
          { idx: 1, tool: "rag_search", description: "search" }
        ]
      })
    )

    await screen.findByText("Pending tool plan")
    const planRoot = screen.getByText("Pending tool plan").closest("div")
    expect(planRoot).not.toBeNull()
    const checkboxes = within(planRoot as HTMLElement).getAllByRole("checkbox")
    fireEvent.click(checkboxes[0])
    fireEvent.click(screen.getByRole("button", { name: "Confirm plan" }))

    await waitFor(() => {
      const sentPayloads = getSentPayloads(ws)
      expect(
        sentPayloads.some(
          (payload) =>
            payload.type === "confirm_plan" &&
            payload.plan_id === "plan-1" &&
            JSON.stringify(payload.approved_steps) === JSON.stringify([1])
        )
      ).toBe(true)
    })

    ws.emitMessage(
      JSON.stringify({
        event: "tool_plan",
        plan_id: "plan-2",
        steps: [{ idx: 0, tool: "summarize", description: "summarize" }]
      })
    )
    await screen.findByText("Pending tool plan")
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))

    await waitFor(() => {
      const sentPayloads = getSentPayloads(ws)
      expect(
        sentPayloads.some(
          (payload) =>
            payload.type === "cancel" &&
            payload.session_id === "sess-12345" &&
            payload.reason === "user_cancelled"
        )
      ).toBe(true)
    })
  })

  it("renders wake controls from the selected profile's saved trigger phrases", async () => {
    mocks.location.search = "?persona_id=research_assistant&tab=live"
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.capabilitiesState.capabilities = {
      hasPersona: true,
      hasPersonalization: true,
      hasAudio: true
    } as any
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant/voice-analytics")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            summary: { total_runs: 0, matched_runs: 0, fallback_runs: 0 }
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: null,
            identity_md: null,
            heartbeat_md: null
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true,
            voice_defaults: {
              voice_chat_trigger_phrases: ["saved wake phrase"],
              wake_behavior: "continuous"
            }
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-memory-status" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    await waitForLiveSessionPanel()

    expect(screen.getByTestId("live-wake-phrases")).toHaveTextContent(
      "saved wake phrase"
    )
    expect(screen.getByTestId("live-wake-behavior")).toHaveTextContent("Continuous")
  })

  it("stops wake listening when leaving the Live tab", async () => {
    const originalSpeechRecognition = (window as any).SpeechRecognition
    class MockSpeechRecognition {
      continuous = false
      interimResults = false
      lang = ""
      onresult: ((event: any) => void) | null = null
      onerror: ((event: any) => void) | null = null
      onend: (() => void) | null = null
      start = vi.fn()
      stop = vi.fn()
    }
    ;(window as any).SpeechRecognition = MockSpeechRecognition

    try {
      mocks.location.search = "?persona_id=research_assistant&tab=live"
      mocks.getConfig.mockResolvedValue({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "persona-key"
      })
      mocks.capabilitiesState.capabilities = {
        hasPersona: true,
        hasPersonalization: true,
        hasAudio: true
      } as any
      mocks.fetchWithAuth.mockImplementation((path: string) => {
        if (path.includes("/persona/catalog")) {
          return Promise.resolve({
            ok: true,
            json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
          })
        }
        if (path.includes("/persona/profiles/research_assistant/voice-analytics")) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              persona_id: "research_assistant",
              summary: { total_runs: 0, matched_runs: 0, fallback_runs: 0 }
            })
          })
        }
        if (path.includes("/persona/profiles/research_assistant/state")) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              persona_id: "research_assistant",
              soul_md: null,
              identity_md: null,
              heartbeat_md: null
            })
          })
        }
        if (path.includes("/persona/profiles/research_assistant")) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "research_assistant",
              use_persona_state_context_default: true,
              voice_defaults: {
                voice_chat_trigger_phrases: ["saved wake phrase"],
                wake_behavior: "continuous"
              }
            })
          })
        }
        if (path.includes("/persona/sessions/sess-tab-switch")) {
          return Promise.resolve({
            ok: true,
            json: async () => ({ preferences: {} })
          })
        }
        if (path === "/api/v1/persona/session") {
          return Promise.resolve({
            ok: true,
            json: async () => ({ session_id: "sess-tab-switch" })
          })
        }
        if (path.includes("/persona/sessions")) {
          return Promise.resolve({
            ok: true,
            json: async () => []
          })
        }
        return Promise.resolve({
          ok: false,
          error: `unhandled path: ${path}`,
          json: async () => ({})
        })
      })

      render(<SidepanelPersona />)

      await waitForLiveSessionPanel()
      fireEvent.click(screen.getByRole("button", { name: "Connect" }))
      await waitFor(() => {
        expect(MockWebSocket.instances).toHaveLength(1)
      })
      const ws = MockWebSocket.instances[0]
      ws.emitOpen()
      await waitFor(() => {
        expect(
          getSentPayloads(ws).some((payload) => payload.type === "voice_config")
        ).toBe(true)
      })

      fireEvent.click(screen.getByTestId("live-wake-toggle"))
      await waitFor(() => {
        expect(screen.getByTestId("live-wake-state")).toHaveTextContent("listening")
      })
      ws.send.mockClear()

      fireEvent.click(screen.getByRole("tab", { name: "Profiles" }))

      await waitFor(() => {
        expect(getSentPayloads(ws)).toEqual(
          expect.arrayContaining([
            expect.objectContaining({
              type: "wake_deactivation",
              reason: "tab_switch"
            })
          ])
        )
      })
    } finally {
      ;(window as any).SpeechRecognition = originalSpeechRecognition
    }
  })

  it("hydrates persisted session preferences when connecting to a resumed session", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/profiles/research_assistant/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: null,
            identity_md: null,
            heartbeat_md: null
          })
        })
      }
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions/sess-pref-hydrated")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: "sess-pref-hydrated",
            preferences: {
              use_memory_context: false,
              use_companion_context: false,
              use_persona_state_context: false,
              memory_top_k: 7
            },
            turns: []
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ session_id: "sess-pref-hydrated" }]
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-pref-hydrated" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)

    fireEvent.click(screen.getByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    await screen.findByText("Persona stream connected")
    await screen.findByText("Memory results: 7")

    await waitFor(() => {
      expect(
        screen.getByTestId("persona-memory-toggle") as HTMLInputElement
      ).not.toBeChecked()
      expect(
        screen.getByTestId("persona-companion-context-toggle") as HTMLInputElement
      ).not.toBeChecked()
      expect(
        screen.getByTestId("persona-state-context-toggle") as HTMLInputElement
      ).not.toBeChecked()
    })
  })

  it("creates companion-mode persona sessions with the companion conversation surface", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-companion" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona mode="companion" />)

    fireEvent.click(screen.getByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(mocks.fetchWithAuth).toHaveBeenCalledWith("/api/v1/persona/session", {
        method: "POST",
        body: {
          persona_id: "research_assistant",
          resume_session_id: undefined,
          surface: "companion.conversation"
        }
      })
    })
  })

  it("filters companion-mode session history to companion conversations", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ session_id: "sess-companion-only" }]
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-companion-only" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona mode="companion" />)

    fireEvent.click(screen.getByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(
        mocks.fetchWithAuth.mock.calls.some(([path]) =>
          String(path).includes(
            "/api/v1/persona/sessions?persona_id=research_assistant&surface=companion.conversation&limit=50"
          )
        )
      ).toBe(true)
    })
  })

  it("prefers tool_result.output and falls back to legacy result alias", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-legacy" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(await screen.findByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })

    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    await screen.findByText("Persona stream connected")

    ws.emitMessage(
      JSON.stringify({
        event: "tool_result",
        step_idx: 2,
        output: "canonical-output",
        result: "legacy-alias"
      })
    )
    await screen.findByText("Result step 2: canonical-output")
    expect(screen.queryByText("Result step 2: legacy-alias")).not.toBeInTheDocument()

    ws.emitMessage(
      JSON.stringify({
        event: "tool_result",
        step_idx: 3,
        result: "legacy-only"
      })
    )
    await screen.findByText("Result step 3: legacy-only")
  })

  it("renders runtime approval requests and retries the tool after approval", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-approval" })
        })
      }
      if (path.includes("/mcp/hub/approval-decisions")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: 101,
            approval_policy_id: init?.body?.approval_policy_id ?? 17,
            context_key: init?.body?.context_key,
            conversation_id: init?.body?.conversation_id,
            tool_name: init?.body?.tool_name,
            scope_key: init?.body?.scope_key,
            decision: init?.body?.decision,
            consume_on_match: init?.body?.duration === "once",
            expires_at: init?.body?.duration === "session" ? "2099-01-01T00:00:00Z" : null
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(await screen.findByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })

    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    ws.emitMessage(
      JSON.stringify({
        event: "tool_result",
        session_id: "sess-approval",
        plan_id: "plan-approval",
        step_idx: 0,
        step_type: "mcp_tool",
        tool: "knowledge.search",
        args: { query: "approval needed" },
        why: "Need to search notes",
        ok: false,
        error: "Runtime approval required",
        reason_code: "APPROVAL_REQUIRED",
        approval: {
          approval_policy_id: 17,
          mode: "ask_outside_profile",
          tool_name: "knowledge.search",
          context_key: "user:1|group:|persona:research_assistant",
          conversation_id: "sess-approval",
          scope_key: "tool:knowledge.search",
          reason: "outside_profile",
          duration_options: ["once", "session"],
          arguments_summary: { query: "approval needed" }
        }
      })
    )

    await screen.findByText("Runtime approval required")
    await screen.findByText("knowledge.search")
    await screen.findByText("outside_profile")

    fireEvent.click(screen.getByRole("button", { name: "Approve and retry" }))

    await waitFor(() => {
      const approvalCall = mocks.fetchWithAuth.mock.calls.find(([path]) =>
        String(path).includes("/mcp/hub/approval-decisions")
      )
      expect(approvalCall).toBeTruthy()
      expect(
        (
          approvalCall?.[1] as {
            body?: {
              approval_policy_id?: number
              tool_name?: string
              decision?: string
              duration?: string
            }
          }
        )?.body
      ).toMatchObject({
        approval_policy_id: 17,
        tool_name: "knowledge.search",
        decision: "approved",
        duration: "once",
      })
      const sentPayloads = getSentPayloads(ws)
      expect(
        sentPayloads.some(
          (payload) =>
            payload.type === "retry_tool_call" &&
            payload.plan_id === "plan-approval" &&
            payload.step_idx === 0 &&
            payload.tool === "knowledge.search"
        )
      ).toBe(true)
    })
  })

  it("renders external runtime approval context with server and slot set", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-ext-approval" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(await screen.findByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })

    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    ws.emitMessage(
      JSON.stringify({
        event: "tool_result",
        session_id: "sess-ext-approval",
        step_idx: 0,
        step_type: "mcp_tool",
        tool: "ext.docs.search",
        args: { query: "approval needed" },
        ok: false,
        error: "Runtime approval required",
        reason_code: "APPROVAL_REQUIRED",
        approval: {
          approval_policy_id: 17,
          mode: "ask_outside_profile",
          tool_name: "ext.docs.search",
          context_key: "user:1|group:|persona:research_assistant",
          conversation_id: "sess-ext-approval",
          scope_key: "tool:ext.docs.search|args:123",
          reason: "external_confirmation_required",
          duration_options: ["once", "session"],
          arguments_summary: { query: "approval needed" },
          scope_context: {
            server_id: "docs",
            requested_slots: ["token_readonly"]
          }
        }
      })
    )

    await screen.findByText("Runtime approval required")
    await screen.findByText("docs")
    await screen.findByText("token_readonly")
  })

  it("renders workspace runtime approval context with workspace and trust source", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-workspace-approval" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(await screen.findByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })

    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    ws.emitMessage(
      JSON.stringify({
        event: "tool_result",
        session_id: "sess-workspace-approval",
        step_idx: 0,
        step_type: "mcp_tool",
        tool: "files.read",
        args: { path: "src/README.md" },
        ok: false,
        error: "Runtime approval required",
        reason_code: "APPROVAL_REQUIRED",
        approval: {
          approval_policy_id: 17,
          mode: "ask_outside_profile",
          tool_name: "files.read",
          context_key: "user:1|group:|persona:research_assistant",
          conversation_id: "sess-workspace-approval",
          scope_key: "tool:files.read|args:123",
          reason: "workspace_not_allowed_but_trusted",
          duration_options: ["once", "session"],
          arguments_summary: { path: "src/README.md" },
          scope_context: {
            workspace_id: "workspace-beta",
            selected_workspace_trust_source: "shared_registry",
            selected_assignment_id: 11
          }
        }
      })
    )

    await screen.findByText("Runtime approval required")
    await screen.findByText("workspace-beta")
    await screen.findByText("shared_registry")
  })

  it("records deny as current-request-only and does not retry the tool", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-deny" })
        })
      }
      if (path.includes("/mcp/hub/approval-decisions")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: 102,
            approval_policy_id: init?.body?.approval_policy_id ?? 17,
            context_key: init?.body?.context_key,
            conversation_id: init?.body?.conversation_id,
            tool_name: init?.body?.tool_name,
            scope_key: init?.body?.scope_key,
            decision: init?.body?.decision,
            consume_on_match: false,
            expires_at: null
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(await screen.findByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })

    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    ws.emitMessage(
      JSON.stringify({
        event: "tool_result",
        session_id: "sess-deny",
        plan_id: "plan-deny",
        step_idx: 0,
        step_type: "mcp_tool",
        tool: "knowledge.search",
        args: { query: "deny me" },
        why: "Need to search notes",
        ok: false,
        error: "Runtime approval required",
        reason_code: "APPROVAL_REQUIRED",
        approval: {
          approval_policy_id: 17,
          mode: "ask_outside_profile",
          tool_name: "knowledge.search",
          context_key: "user:1|group:|persona:research_assistant",
          conversation_id: "sess-deny",
          scope_key: "tool:knowledge.search",
          reason: "outside_profile",
          duration_options: ["once", "session"],
          arguments_summary: { query: "deny me" }
        }
      })
    )

    await screen.findByText("Runtime approval required")
    fireEvent.click(screen.getByRole("button", { name: "Deny" }))

    await waitFor(() => {
      const approvalCall = mocks.fetchWithAuth.mock.calls.find(([path, init]) =>
        String(path).includes("/mcp/hub/approval-decisions") &&
        init?.body?.decision === "denied"
      )
      expect(approvalCall).toBeTruthy()
      const body = (
        approvalCall?.[1] as {
          body?: {
            decision?: string
            duration?: string
          }
        }
      )?.body
      expect(body?.decision).toBe("denied")
      expect(body?.duration).toBe("once")
      const sentPayloads = getSentPayloads(ws)
      expect(sentPayloads.some((payload) => payload.type === "retry_tool_call")).toBe(false)
    })
  })

  it.each([
    {
      reasonCode: "required_slot_not_granted",
      label: "Credential slots not granted: token_readonly",
      payload: {
        external_access: {
          server_id: "docs",
          missing_bound_slots: ["token_readonly"],
          blocked_reason: "required_slot_not_granted"
        }
      }
    },
    {
      reasonCode: "required_slot_secret_missing",
      label: "Credential secrets missing: token_readonly",
      payload: {
        external_access: {
          server_id: "docs",
          missing_secret_slots: ["token_readonly"],
          blocked_reason: "required_slot_secret_missing"
        }
      }
    }
  ])(
    "renders explicit hard-deny external slot messaging for $reasonCode",
    async ({ reasonCode, label, payload }) => {
      mocks.getConfig.mockResolvedValue({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "persona-key"
      })
      mocks.fetchWithAuth.mockImplementation((path: string) => {
        if (path.includes("/persona/catalog")) {
          return Promise.resolve({
            ok: true,
            json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
          })
        }
        if (path.includes("/persona/sessions")) {
          return Promise.resolve({
            ok: true,
            json: async () => []
          })
        }
        if (path.includes("/persona/session")) {
          return Promise.resolve({
            ok: true,
            json: async () => ({ session_id: "sess-ext-deny" })
          })
        }
        return Promise.resolve({
          ok: false,
          error: `unhandled path: ${path}`,
          json: async () => ({})
        })
      })

      render(<SidepanelPersona />)
      fireEvent.click(screen.getByRole("button", { name: "Connect" }))

      await waitFor(() => {
        expect(MockWebSocket.instances).toHaveLength(1)
      })

      const ws = MockWebSocket.instances[0]
      ws.emitOpen()

      ws.emitMessage(
        JSON.stringify({
          event: "tool_result",
          session_id: "sess-ext-deny",
          step_idx: 0,
          step_type: "mcp_tool",
          tool: "ext.docs.search",
          args: { query: "blocked" },
          ok: false,
          error: "Blocked external credential use",
          reason_code: reasonCode,
          ...payload
        })
      )

      await screen.findByText(label)
      expect(screen.queryByRole("button", { name: "Approve and retry" })).not.toBeInTheDocument()
    }
  )

  it("renders explicit hard-deny workspace trust-source messaging without approval controls", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-workspace-deny" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })

    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    ws.emitMessage(
      JSON.stringify({
        event: "tool_result",
        session_id: "sess-workspace-deny",
        step_idx: 0,
        step_type: "mcp_tool",
        tool: "files.read",
        args: { path: "src/README.md" },
        ok: false,
        error: "Blocked path-scoped tool use",
        reason_code: "workspace_unresolvable_for_trust_source",
        path_scope: {
          workspace_id: "workspace-missing",
          selected_workspace_trust_source: "shared_registry",
          reason: "workspace_unresolvable_for_trust_source"
        }
      })
    )

    await screen.findByText("Blocked: workspace is not resolvable through the required trust source.")
    expect(screen.queryByRole("button", { name: "Approve and retry" })).not.toBeInTheDocument()
  })

  it("renders explicit hard-deny multi-root ambiguity messaging without approval controls", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-multiroot-deny" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })

    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    ws.emitMessage(
      JSON.stringify({
        event: "tool_result",
        session_id: "sess-multiroot-deny",
        step_idx: 0,
        step_type: "mcp_tool",
        tool: "files.read",
        args: { paths: ["/tmp/workspace-alpha/docs/shared.md"] },
        ok: false,
        error: "Blocked path-scoped tool use",
        reason_code: "path_matches_multiple_workspace_roots",
        path_scope: {
          workspace_bundle_ids: ["workspace-alpha", "workspace-alpha-docs"],
          normalized_paths: ["/tmp/workspace-alpha/docs/shared.md"],
          reason: "path_matches_multiple_workspace_roots"
        }
      })
    )

    await screen.findByText("Blocked: path matched multiple trusted workspace roots.")
    expect(screen.queryByRole("button", { name: "Approve and retry" })).not.toBeInTheDocument()
  })

  it("renders multi-root approval context with exact workspace bundle and path set", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-multiroot-approval" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })

    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    ws.emitMessage(
      JSON.stringify({
        event: "tool_result",
        session_id: "sess-multiroot-approval",
        step_idx: 0,
        step_type: "mcp_tool",
        tool: "files.read",
        args: {
          paths: [
            "/tmp/workspace-alpha/src/README.md",
            "/tmp/workspace-beta/docs/index.md"
          ]
        },
        ok: false,
        error: "Runtime approval required",
        reason_code: "APPROVAL_REQUIRED",
        approval: {
          approval_policy_id: 17,
          mode: "ask_outside_profile",
          tool_name: "files.read",
          context_key: "user:1|group:|persona:research_assistant",
          conversation_id: "sess-multiroot-approval",
          scope_key: "tool:files.read|args:456",
          reason: "path_outside_allowlist_scope",
          duration_options: ["once", "session"],
          arguments_summary: {
            paths: [
              "/tmp/workspace-alpha/src/README.md",
              "/tmp/workspace-beta/docs/index.md"
            ]
          },
          scope_context: {
            workspace_bundle_ids: ["workspace-alpha", "workspace-beta"],
            normalized_paths: [
              "/tmp/workspace-alpha/src/README.md",
              "/tmp/workspace-beta/docs/index.md"
            ],
            reason: "path_outside_allowlist_scope"
          }
        }
      })
    )

    await screen.findByText("Runtime approval required")
    await screen.findByText("workspace-alpha")
    await screen.findByText("workspace-beta")
    await screen.findByText("/tmp/workspace-alpha/src/README.md")
    await screen.findByText("/tmp/workspace-beta/docs/index.md")
  })

  it("renders policy metadata and keeps blocked steps out of approvals", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-policy" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })

    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    ws.emitMessage(
      JSON.stringify({
        event: "tool_plan",
        plan_id: "plan-policy",
        steps: [
          {
            idx: 0,
            tool: "export_report",
            description: "Export report",
            policy: {
              allow: false,
              requires_confirmation: true,
              required_scope: "write:export",
              reason_code: "POLICY_EXPORT_DISABLED",
              reason: "Export tools are disabled by persona policy."
            }
          },
          {
            idx: 1,
            tool: "rag_search",
            description: "Search notes",
            policy: {
              allow: true,
              requires_confirmation: false,
              required_scope: "read"
            }
          }
        ]
      })
    )

    await screen.findByText("Pending tool plan")
    await screen.findByText("scope: write:export")
    await screen.findByText("blocked: POLICY_EXPORT_DISABLED")
    await screen.findByText("Export tools are disabled by persona policy.")

    const planRoot = screen.getByText("Pending tool plan").closest("div")
    expect(planRoot).not.toBeNull()
    const checkboxes = within(planRoot as HTMLElement).getAllByRole("checkbox")
    expect(checkboxes[0]).toBeDisabled()
    expect(checkboxes[0]).not.toBeChecked()
    expect(checkboxes[1]).toBeChecked()

    fireEvent.click(screen.getByRole("button", { name: "Confirm plan" }))
    await waitFor(() => {
      const sentPayloads = getSentPayloads(ws)
      expect(
        sentPayloads.some(
          (payload) =>
            payload.type === "confirm_plan" &&
            payload.plan_id === "plan-policy" &&
            JSON.stringify(payload.approved_steps) === JSON.stringify([1])
        )
      ).toBe(true)
    })
  })

  it("sends per-message memory controls in user_message payload", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-memory-controls" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })

    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    await screen.findByText("Memory results: 3")
    expect(screen.queryByText("k=3")).not.toBeInTheDocument()

    ws.emitMessage(
      JSON.stringify({
        event: "tool_plan",
        plan_id: "plan-memory-labels",
        memory: {
          enabled: true,
          requested_top_k: 4,
          applied_count: 2
        },
        companion: {
          enabled: true,
          requested_enabled: true,
          applied_card_count: 1,
          applied_activity_count: 2
        },
        steps: [{ idx: 0, tool: "rag_search", description: "search" }]
      })
    )
    await screen.findByText("requested memory results: 4")
    await screen.findByText("applied results: 2")
    await screen.findByText("companion on")
    await screen.findByText("applied cards: 1")
    await screen.findByText("applied activity: 2")

    fireEvent.click(screen.getByTestId("persona-memory-toggle"))
    fireEvent.click(screen.getByTestId("persona-state-context-toggle"))
    fireEvent.click(screen.getByTestId("persona-companion-context-toggle"))
    fireEvent.change(screen.getByPlaceholderText("Ask Persona..."), {
      target: { value: "memory toggle payload" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))

    await waitFor(() => {
      const sentPayloads = getSentPayloads(ws)
      const userMessage = sentPayloads.find((payload) => payload.type === "user_message")
      expect(userMessage).toBeTruthy()
      expect(userMessage?.use_memory_context).toBe(false)
      expect(userMessage?.use_persona_state_context).toBe(false)
      expect(userMessage?.use_companion_context).toBe(false)
      expect(userMessage?.memory_top_k).toBe(3)
    })
  })

  it("shows live memory mode status from the selected persona and controls", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            {
              id: "research_assistant",
              name: "Research Assistant",
              mode: "persistent_scoped"
            }
          ]
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(await screen.findByRole("button", { name: "Connect" }))

    const memoryStatus = await screen.findByTestId("persona-memory-status")
    await waitFor(() => {
      expect(memoryStatus).toHaveTextContent("Mode: persistent scoped")
    })
    expect(memoryStatus).toHaveTextContent("Memory: on")
    expect(memoryStatus).toHaveTextContent("Top-k: 3")
    expect(memoryStatus).toHaveTextContent("State context: available")

    fireEvent.click(screen.getByTestId("persona-memory-toggle"))

    expect(memoryStatus).toHaveTextContent("Memory: off")
  })

  it("updates persona state-context profile default and applies it to outgoing messages", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      if (path.includes("/persona/profiles/research_assistant/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: null,
            identity_md: null,
            heartbeat_md: null
          })
        })
      }
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        if (String(init?.method || "GET").toUpperCase() === "PATCH") {
          const nextDefault = Boolean(init?.body?.use_persona_state_context_default)
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "research_assistant",
              use_persona_state_context_default: nextDefault
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: false
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-state-default" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    const ws = MockWebSocket.instances[0]
    ws.emitOpen()

    const stateToggleInput = screen.getByTestId(
      "persona-state-context-toggle"
    ) as HTMLInputElement
    const stateDefaultToggleInput = screen.getByTestId(
      "persona-state-context-default-toggle"
    ) as HTMLInputElement

    await waitFor(() => {
      expect(stateToggleInput).not.toBeChecked()
      expect(stateDefaultToggleInput).not.toBeChecked()
      expect(stateDefaultToggleInput).not.toBeDisabled()
    })

    await waitForLiveSessionPanel()
    fireEvent.click(screen.getByTestId("persona-state-context-default-toggle"))

    await waitFor(() => {
      const patchCall = mocks.fetchWithAuth.mock.calls.find(
        ([calledPath, calledInit]) =>
          String(calledPath).includes("/persona/profiles/research_assistant") &&
          String((calledInit as { method?: string } | undefined)?.method || "").toUpperCase() ===
            "PATCH"
      )
      expect(patchCall).toBeTruthy()
      expect(
        (patchCall?.[1] as { body?: { use_persona_state_context_default?: boolean } } | undefined)
          ?.body
      ).toEqual({ use_persona_state_context_default: true })
      expect(stateToggleInput).toBeChecked()
      expect(stateDefaultToggleInput).toBeChecked()
    })

    fireEvent.change(screen.getByPlaceholderText("Ask Persona..."), {
      target: { value: "state default payload" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))

    await waitFor(() => {
      const sentPayloads = getSentPayloads(ws)
      const userMessage = sentPayloads.find((payload) => payload.type === "user_message")
      expect(userMessage).toBeTruthy()
      expect(userMessage?.use_persona_state_context).toBe(true)
    })
  })

  it("loads, saves, and restores persona state docs from sidepanel controls", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state/history")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            entries: [
              {
                entry_id: "hist-1",
                field: "soul_md",
                content: "archived soul version",
                is_active: false,
                version: 1
              }
            ]
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state/restore")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: "restored soul version",
            identity_md: "initial identity",
            heartbeat_md: "initial heartbeat"
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state")) {
        if (method === "PUT") {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              persona_id: "research_assistant",
              soul_md: init?.body?.soul_md ?? null,
              identity_md: init?.body?.identity_md ?? null,
              heartbeat_md: init?.body?.heartbeat_md ?? null,
              last_modified: "2026-02-22T08:00:00Z"
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: "initial soul",
            identity_md: "initial identity",
            heartbeat_md: "initial heartbeat",
            last_modified: "2026-02-22T07:00:00Z"
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-state-docs" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))
    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    MockWebSocket.instances[0].emitOpen()
    await openStateDocsTab()

    const soulInput = await waitForStateDocsEditor()
    const identityInput = screen.getByTestId(
      "persona-state-identity-input"
    ) as HTMLTextAreaElement
    const heartbeatInput = screen.getByTestId(
      "persona-state-heartbeat-input"
    ) as HTMLTextAreaElement

    await waitFor(() => {
      expect(soulInput.value).toBe("initial soul")
      expect(identityInput.value).toBe("initial identity")
      expect(heartbeatInput.value).toBe("initial heartbeat")
    })

    fireEvent.change(soulInput, { target: { value: "updated soul draft" } })
    fireEvent.click(screen.getByTestId("persona-state-save-button"))

    await waitFor(() => {
      const putCall = mocks.fetchWithAuth.mock.calls.find(
        ([calledPath, calledInit]) =>
          String(calledPath).includes("/persona/profiles/research_assistant/state") &&
          String((calledInit as { method?: string } | undefined)?.method || "").toUpperCase() ===
            "PUT"
      )
      expect(putCall).toBeTruthy()
      expect(
        (putCall?.[1] as { body?: { soul_md?: string } } | undefined)?.body?.soul_md
      ).toBe("updated soul draft")
    })

    fireEvent.click(screen.getByTestId("persona-state-history-button"))
    await screen.findByText("archived soul version")

    fireEvent.click(screen.getByTestId("persona-state-restore-hist-1"))
    await waitFor(() => {
      const restoreCall = mocks.fetchWithAuth.mock.calls.find(
        ([calledPath, calledInit]) =>
          String(calledPath).includes("/persona/profiles/research_assistant/state/restore") &&
          String((calledInit as { method?: string } | undefined)?.method || "").toUpperCase() ===
            "POST"
      )
      expect(restoreCall).toBeTruthy()
      expect(
        (
          restoreCall?.[1] as { body?: { entry_id?: string } } | undefined
        )?.body?.entry_id
      ).toBe("hist-1")
      expect(soulInput.value).toBe("restored soul version")
    })
  })

  it("targets catalog-resolved persona for profile and state writes when connected", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "builder_bot", name: "Builder Bot" }]
        })
      }
      if (path.includes("/persona/profiles/builder_bot/state")) {
        if (method === "PUT") {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              persona_id: "builder_bot",
              soul_md: init?.body?.soul_md ?? null,
              identity_md: init?.body?.identity_md ?? null,
              heartbeat_md: init?.body?.heartbeat_md ?? null
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "builder_bot",
            soul_md: "builder soul",
            identity_md: "builder identity",
            heartbeat_md: "builder heartbeat"
          })
        })
      }
      if (path.includes("/persona/profiles/builder_bot")) {
        if (method === "PATCH") {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "builder_bot",
              use_persona_state_context_default: Boolean(
                init?.body?.use_persona_state_context_default
              )
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "builder_bot",
            use_persona_state_context_default: false
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: "sess-builder",
            persona: { id: "builder_bot" }
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(await screen.findByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    MockWebSocket.instances[0].emitOpen()

    await waitFor(() => {
      const sessionCall = mocks.fetchWithAuth.mock.calls.find(([calledPath]) =>
        String(calledPath).includes("/persona/sessions?persona_id=builder_bot")
      )
      expect(sessionCall).toBeTruthy()
      const profileGetCall = mocks.fetchWithAuth.mock.calls.find(
        ([calledPath, calledInit]) =>
          String(calledPath).includes("/persona/profiles/builder_bot") &&
          String((calledInit as { method?: string } | undefined)?.method || "").toUpperCase() ===
            "GET"
      )
      expect(profileGetCall).toBeTruthy()
    })

    await waitForLiveSessionPanel()
    fireEvent.click(screen.getByTestId("persona-state-context-default-toggle"))
    await openStateDocsTab()
    const soulInput = await waitForStateDocsEditor()
    fireEvent.change(soulInput, {
      target: { value: "builder state update" }
    })
    fireEvent.click(screen.getByTestId("persona-state-save-button"))

    await waitFor(() => {
      const profilePatchCall = mocks.fetchWithAuth.mock.calls.find(
        ([calledPath, calledInit]) =>
          String(calledPath).includes("/persona/profiles/builder_bot") &&
          String((calledInit as { method?: string } | undefined)?.method || "").toUpperCase() ===
            "PATCH"
      )
      expect(profilePatchCall).toBeTruthy()
      const statePutCall = mocks.fetchWithAuth.mock.calls.find(
        ([calledPath, calledInit]) =>
          String(calledPath).includes("/persona/profiles/builder_bot/state") &&
          String((calledInit as { method?: string } | undefined)?.method || "").toUpperCase() ===
            "PUT"
      )
      expect(statePutCall).toBeTruthy()
      expect(
        String(profilePatchCall?.[0]).includes("/persona/profiles/research_assistant")
      ).toBe(false)
      expect(
        String(statePutCall?.[0]).includes("/persona/profiles/research_assistant/state")
      ).toBe(false)
    })
  })

  it("tracks dirty state and supports reverting state docs without saving", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: "stable soul",
            identity_md: "stable identity",
            heartbeat_md: "stable heartbeat"
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-dirty-revert" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    MockWebSocket.instances[0].emitOpen()
    await openStateDocsTab()

    const soulInput = await waitForStateDocsEditor()
    const dirtyTag = screen.getByTestId("persona-state-dirty-tag")
    const saveButton = screen.getByTestId("persona-state-save-button")
    const revertButton = screen.getByTestId("persona-state-revert-button")

    await waitFor(() => {
      expect(soulInput.value).toBe("stable soul")
      expect(dirtyTag).toHaveTextContent("saved")
      expect(saveButton).toBeDisabled()
      expect(revertButton).toBeDisabled()
    })

    fireEvent.change(soulInput, { target: { value: "edited soul" } })
    await waitFor(() => {
      expect(dirtyTag).toHaveTextContent("unsaved")
      expect(saveButton).not.toBeDisabled()
      expect(revertButton).not.toBeDisabled()
    })

    fireEvent.click(revertButton)
    await waitFor(() => {
      expect(soulInput.value).toBe("stable soul")
      expect(dirtyTag).toHaveTextContent("saved")
      expect(saveButton).toBeDisabled()
      expect(revertButton).toBeDisabled()
    })

    const putCall = mocks.fetchWithAuth.mock.calls.find(
      ([calledPath, calledInit]) =>
        String(calledPath).includes("/persona/profiles/research_assistant/state") &&
        String((calledInit as { method?: string } | undefined)?.method || "").toUpperCase() ===
          "PUT"
    )
    expect(putCall).toBeUndefined()
  })

  it("registers and removes beforeunload guard based on draft dirty state", async () => {
    const addEventListenerSpy = vi.spyOn(window, "addEventListener")
    const removeEventListenerSpy = vi.spyOn(window, "removeEventListener")
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: "stable soul",
            identity_md: "stable identity",
            heartbeat_md: "stable heartbeat"
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-beforeunload-guard" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    MockWebSocket.instances[0].emitOpen()
    await openStateDocsTab()

    const soulInput = await waitForStateDocsEditor()
    const revertButton = screen.getByTestId("persona-state-revert-button")
    await waitFor(() => {
      expect(soulInput.value).toBe("stable soul")
    })

    fireEvent.change(soulInput, { target: { value: "unsaved beforeunload draft" } })

    let beforeUnloadHandler: ((event: BeforeUnloadEvent) => string | undefined) | null = null
    await waitFor(() => {
      const beforeUnloadCall = addEventListenerSpy.mock.calls.find(
        ([eventName]) => eventName === "beforeunload"
      )
      expect(beforeUnloadCall).toBeTruthy()
      beforeUnloadHandler = beforeUnloadCall?.[1] as
        | ((event: BeforeUnloadEvent) => string | undefined)
        | null
      expect(beforeUnloadHandler).toBeTruthy()
    })

    const beforeUnloadEvent = new Event("beforeunload", { cancelable: true }) as BeforeUnloadEvent
    const beforeUnloadResult = beforeUnloadHandler?.(beforeUnloadEvent)
    expect(beforeUnloadEvent.defaultPrevented).toBe(true)
    expect(String(beforeUnloadResult || "")).toContain(
      "You have unsaved state-doc changes"
    )
    expect(String(beforeUnloadResult || "")).toContain("Leave this page without saving")

    fireEvent.click(revertButton)
    await waitFor(() => {
      expect(soulInput.value).toBe("stable soul")
      const removeCall = removeEventListenerSpy.mock.calls.find(
        ([eventName, handler]) =>
          eventName === "beforeunload" && handler === beforeUnloadHandler
      )
      expect(removeCall).toBeTruthy()
    })

    addEventListenerSpy.mockRestore()
    removeEventListenerSpy.mockRestore()
  })

  it("resets blocked in-app navigation when unsaved draft discard is declined", async () => {
    const confirmSpy = vi.spyOn(window, "confirm")
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: "stable soul",
            identity_md: "stable identity",
            heartbeat_md: "stable heartbeat"
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-router-block-decline" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))
    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    MockWebSocket.instances[0].emitOpen()
    await openStateDocsTab()

    const soulInput = await waitForStateDocsEditor()
    await waitFor(() => {
      expect(soulInput.value).toBe("stable soul")
    })
    fireEvent.change(soulInput, { target: { value: "dirty route blocker draft" } })

    await waitFor(() => {
      expect(mocks.useBlocker.mock.calls.some(([when]) => when === true)).toBe(true)
    })

    confirmSpy.mockClear()
    mocks.blocker.proceed.mockClear()
    mocks.blocker.reset.mockClear()
    mocks.blocker.state = "blocked"
    confirmSpy.mockReturnValueOnce(false)
    fireEvent.change(soulInput, { target: { value: "dirty route blocker draft v2" } })

    await waitFor(() => {
      expect(confirmSpy).toHaveBeenCalled()
      expect(mocks.blocker.reset).toHaveBeenCalledTimes(1)
      expect(mocks.blocker.proceed).not.toHaveBeenCalled()
    })
    expect(String(confirmSpy.mock.calls[0]?.[0] || "")).toContain(
      "Leave this page and discard local drafts"
    )
    confirmSpy.mockRestore()
  })

  it("proceeds blocked in-app navigation when unsaved draft discard is confirmed", async () => {
    const confirmSpy = vi.spyOn(window, "confirm")
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: "stable soul",
            identity_md: "stable identity",
            heartbeat_md: "stable heartbeat"
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-router-block-confirm" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))
    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    MockWebSocket.instances[0].emitOpen()
    await openStateDocsTab()

    const soulInput = await waitForStateDocsEditor()
    await waitFor(() => {
      expect(soulInput.value).toBe("stable soul")
    })
    fireEvent.change(soulInput, { target: { value: "dirty route blocker confirm draft" } })

    await waitFor(() => {
      expect(mocks.useBlocker.mock.calls.some(([when]) => when === true)).toBe(true)
    })

    confirmSpy.mockClear()
    mocks.blocker.proceed.mockClear()
    mocks.blocker.reset.mockClear()
    mocks.blocker.state = "blocked"
    confirmSpy.mockReturnValueOnce(true)
    fireEvent.change(soulInput, { target: { value: "dirty route blocker confirm draft v2" } })

    await waitFor(() => {
      expect(confirmSpy).toHaveBeenCalled()
      expect(mocks.blocker.proceed).toHaveBeenCalledTimes(1)
      expect(mocks.blocker.reset).not.toHaveBeenCalled()
    })
    expect(String(confirmSpy.mock.calls[0]?.[0] || "")).toContain(
      "Leave this page and discard local drafts"
    )
    confirmSpy.mockRestore()
  })

  it("refreshes loaded state history after saving updated state docs", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    let historyFetchCount = 0
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = String(init?.method || "GET").toUpperCase()
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state/history")) {
        historyFetchCount += 1
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            entries: [
              {
                entry_id: "hist-refresh",
                field: "soul_md",
                content: "history value",
                is_active: false,
                version: historyFetchCount
              }
            ]
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state")) {
        if (method === "PUT") {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              persona_id: "research_assistant",
              soul_md: init?.body?.soul_md ?? null,
              identity_md: init?.body?.identity_md ?? null,
              heartbeat_md: init?.body?.heartbeat_md ?? null
            })
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: "history base soul",
            identity_md: "history base identity",
            heartbeat_md: "history base heartbeat"
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-history-refresh" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))
    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    MockWebSocket.instances[0].emitOpen()
    await screen.findByText("Persona stream connected")
    await openStateDocsTab()

    fireEvent.click(screen.getByTestId("persona-state-history-button"))
    await waitFor(() => {
      expect(historyFetchCount).toBe(1)
    })

    fireEvent.change(screen.getByTestId("persona-state-soul-input"), {
      target: { value: "history refresh update" }
    })
    fireEvent.click(screen.getByTestId("persona-state-save-button"))

    await waitFor(() => {
      expect(historyFetchCount).toBeGreaterThanOrEqual(2)
    })
  })

  it("orders state history entries and displays metadata", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state/history")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            entries: [
              {
                entry_id: "older-entry",
                field: "soul_md",
                content: "older history content",
                is_active: false,
                version: 1,
                created_at: "2026-02-20T01:00:00Z",
                last_modified: "2026-02-20T01:10:00Z"
              },
              {
                entry_id: "newer-entry",
                field: "soul_md",
                content: "newer history content",
                is_active: false,
                version: 2,
                created_at: "2026-02-21T01:00:00Z",
                last_modified: "2026-02-21T01:10:00Z"
              }
            ]
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: "base soul",
            identity_md: "base identity",
            heartbeat_md: "base heartbeat"
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-history-order" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))
    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    MockWebSocket.instances[0].emitOpen()
    await screen.findByText("Persona stream connected")
    await openStateDocsTab()

    fireEvent.click(screen.getByTestId("persona-state-history-button"))
    await screen.findByText("newer history content")
    await screen.findByText("older history content")

    const newestCards = screen.getAllByTestId(/persona-state-history-entry-/)
    expect(newestCards[0]).toHaveTextContent("newer history content")
    expect(screen.getByTestId("persona-state-history-meta-newer-entry")).toHaveTextContent(
      "created 2026-02-21T01:00:00Z"
    )
    expect(screen.getByTestId("persona-state-history-meta-newer-entry")).toHaveTextContent(
      "updated 2026-02-21T01:10:00Z"
    )

    fireEvent.click(screen.getByTestId("persona-state-history-order-oldest-button"))
    const oldestCards = screen.getAllByTestId(/persona-state-history-entry-/)
    expect(oldestCards[0]).toHaveTextContent("older history content")
  })

  it("restores and persists state history/editor preferences via localStorage", async () => {
    window.localStorage.setItem("sidepanel:persona:state-editor-expanded", "false")
    window.localStorage.setItem("sidepanel:persona:state-history-order", "oldest")

    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state/history")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            entries: [
              {
                entry_id: "pref-older",
                field: "soul_md",
                content: "pref older content",
                is_active: false,
                version: 1,
                created_at: "2026-02-20T01:00:00Z"
              },
              {
                entry_id: "pref-newer",
                field: "soul_md",
                content: "pref newer content",
                is_active: false,
                version: 2,
                created_at: "2026-02-21T01:00:00Z"
              }
            ]
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: "pref soul",
            identity_md: "pref identity",
            heartbeat_md: "pref heartbeat"
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-pref-persist" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))
    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    MockWebSocket.instances[0].emitOpen()
    await screen.findByText("Persona stream connected")
    await openStateDocsTab()

    expect(screen.queryByTestId("persona-state-soul-input")).not.toBeInTheDocument()
    expect(screen.getByText("Show editor")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("persona-state-editor-toggle-button"))
    expect(screen.getByTestId("persona-state-soul-input")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("persona-state-history-button"))
    await screen.findByText("pref older content")
    await screen.findByText("pref newer content")

    const oldestDefaultCards = screen.getAllByTestId(/persona-state-history-entry-/)
    expect(oldestDefaultCards[0]).toHaveTextContent("pref older content")

    fireEvent.click(screen.getByTestId("persona-state-history-order-newest-button"))
    const newestCards = screen.getAllByTestId(/persona-state-history-entry-/)
    expect(newestCards[0]).toHaveTextContent("pref newer content")
    expect(window.localStorage.getItem("sidepanel:persona:state-history-order")).toBe(
      "newest"
    )

    fireEvent.click(screen.getByTestId("persona-state-editor-toggle-button"))
    expect(window.localStorage.getItem("sidepanel:persona:state-editor-expanded")).toBe(
      "false"
    )
  })

  it("blocks connect when unsaved drafts exist and discard prompt is declined", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false)
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: "server soul",
            identity_md: "server identity",
            heartbeat_md: "server heartbeat"
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-connect-guard" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    const initialCreateSessionCalls = mocks.fetchWithAuth.mock.calls.filter(
      ([path]) => String(path) === "/api/v1/persona/session"
    ).length
    await openStateDocsTab()
    fireEvent.change(await waitForStateDocsEditor(), {
      target: { value: "local draft soul" }
    })
    await openPersonaTab("Live Session")
    fireEvent.click(await screen.findByRole("button", { name: "Connect" }))

    await waitFor(() => {
      expect(confirmSpy).toHaveBeenCalled()
      expect(MockWebSocket.instances).toHaveLength(0)
    })
    const createSessionCalls = mocks.fetchWithAuth.mock.calls.filter(
      ([path]) => String(path) === "/api/v1/persona/session"
    )
    expect(createSessionCalls).toHaveLength(initialCreateSessionCalls)
    expect(String(confirmSpy.mock.calls[0]?.[0] || "")).toContain(
      "Connect and discard local drafts"
    )
    confirmSpy.mockRestore()
  })

  it("keeps session connected when unsaved drafts exist and disconnect prompt is declined", async () => {
    const confirmSpy = vi.spyOn(window, "confirm")
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: "connected soul",
            identity_md: "connected identity",
            heartbeat_md: "connected heartbeat"
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-disconnect-guard" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))
    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    MockWebSocket.instances[0].emitOpen()
    await screen.findByText("Persona stream connected")
    await openStateDocsTab()

    fireEvent.change(await waitForStateDocsEditor(), {
      target: { value: "unsaved disconnect draft" }
    })

    confirmSpy.mockClear()
    confirmSpy.mockReturnValueOnce(false)
    await openPersonaTab("Live Session")
    fireEvent.click(await screen.findByRole("button", { name: /Disconnect/i }))
    await waitFor(() => {
      expect(confirmSpy).toHaveBeenCalled()
    })
    expect(String(confirmSpy.mock.calls[0]?.[0] || "")).toContain(
      "Disconnect and discard local drafts"
    )
    expect(screen.getByRole("button", { name: /Disconnect/i })).toBeInTheDocument()

    confirmSpy.mockReturnValueOnce(true)
    fireEvent.click(screen.getByRole("button", { name: /Disconnect/i }))
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /^(?:loading)?Connect$/i })
      ).toBeInTheDocument()
    })
    confirmSpy.mockRestore()
  })

  it("blocks restore when unsaved drafts exist until discard is confirmed", async () => {
    const confirmSpy = vi.spyOn(window, "confirm")
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation(
      (path: string, init?: { method?: string; body?: Record<string, unknown> }) => {
        const method = String(init?.method || "GET").toUpperCase()
        if (path.includes("/persona/catalog")) {
          return Promise.resolve({
            ok: true,
            json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
          })
        }
        if (path.includes("/persona/profiles/research_assistant/state/history")) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              persona_id: "research_assistant",
              entries: [
                {
                  entry_id: "restore-guard-entry",
                  field: "soul_md",
                  content: "history soul version",
                  is_active: false,
                  version: 1
                }
              ]
            })
          })
        }
        if (path.includes("/persona/profiles/research_assistant/state/restore")) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              persona_id: "research_assistant",
              soul_md: "restored from history",
              identity_md: "initial identity",
              heartbeat_md: "initial heartbeat"
            })
          })
        }
        if (path.includes("/persona/profiles/research_assistant/state")) {
          if (method === "PUT") {
            return Promise.resolve({
              ok: true,
              json: async () => ({
                persona_id: "research_assistant",
                soul_md: init?.body?.soul_md ?? null,
                identity_md: init?.body?.identity_md ?? null,
                heartbeat_md: init?.body?.heartbeat_md ?? null
              })
            })
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              persona_id: "research_assistant",
              soul_md: "initial soul",
              identity_md: "initial identity",
              heartbeat_md: "initial heartbeat"
            })
          })
        }
        if (path.includes("/persona/profiles/research_assistant")) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              id: "research_assistant",
              use_persona_state_context_default: true
            })
          })
        }
        if (path.includes("/persona/sessions")) {
          return Promise.resolve({
            ok: true,
            json: async () => []
          })
        }
        if (path.includes("/persona/session")) {
          return Promise.resolve({
            ok: true,
            json: async () => ({ session_id: "sess-restore-guard" })
          })
        }
        return Promise.resolve({
          ok: false,
          error: `unhandled path: ${path}`,
          json: async () => ({})
        })
      }
    )

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))
    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    MockWebSocket.instances[0].emitOpen()
    await screen.findByText("Persona stream connected")
    await openStateDocsTab()

    const soulInput = await waitForStateDocsEditor()
    await waitFor(() => {
      expect(soulInput.value).toBe("initial soul")
    })
    fireEvent.click(screen.getByTestId("persona-state-history-button"))
    await screen.findByText("history soul version")

    fireEvent.change(soulInput, {
      target: { value: "unsaved local draft before restore" }
    })

    confirmSpy.mockClear()
    confirmSpy.mockReturnValueOnce(false)
    fireEvent.click(screen.getByTestId("persona-state-restore-restore-guard-entry"))
    await waitFor(() => {
      expect(confirmSpy).toHaveBeenCalled()
    })
    expect(String(confirmSpy.mock.calls[0]?.[0] || "")).toContain(
      "Restore this state version and discard local drafts"
    )
    const declinedRestoreCall = mocks.fetchWithAuth.mock.calls.find(
      ([calledPath, calledInit]) =>
        String(calledPath).includes("/persona/profiles/research_assistant/state/restore") &&
        String((calledInit as { method?: string } | undefined)?.method || "").toUpperCase() ===
          "POST"
    )
    expect(declinedRestoreCall).toBeFalsy()
    expect(soulInput.value).toBe("unsaved local draft before restore")

    confirmSpy.mockReturnValueOnce(true)
    fireEvent.click(screen.getByTestId("persona-state-restore-restore-guard-entry"))
    await waitFor(() => {
      const acceptedRestoreCall = mocks.fetchWithAuth.mock.calls.find(
        ([calledPath, calledInit]) =>
          String(calledPath).includes("/persona/profiles/research_assistant/state/restore") &&
          String((calledInit as { method?: string } | undefined)?.method || "").toUpperCase() ===
            "POST"
      )
      expect(acceptedRestoreCall).toBeTruthy()
      expect(soulInput.value).toBe("restored from history")
    })
    confirmSpy.mockRestore()
  })

  it("shows empty history message and supports state editor collapse/expand", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "persona-key"
    })
    mocks.fetchWithAuth.mockImplementation((path: string) => {
      if (path.includes("/persona/catalog")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ id: "research_assistant", name: "Research Assistant" }]
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state/history")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            entries: []
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            persona_id: "research_assistant",
            soul_md: "state soul",
            identity_md: "state identity",
            heartbeat_md: "state heartbeat"
          })
        })
      }
      if (path.includes("/persona/profiles/research_assistant")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "research_assistant",
            use_persona_state_context_default: true
          })
        })
      }
      if (path.includes("/persona/sessions")) {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path.includes("/persona/session")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ session_id: "sess-empty-history" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<SidepanelPersona />)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))
    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1)
    })
    MockWebSocket.instances[0].emitOpen()
    await screen.findByText("Persona stream connected")
    await openStateDocsTab()

    fireEvent.click(screen.getByTestId("persona-state-history-button"))
    await screen.findByTestId("persona-state-history-empty")
    expect(screen.getByText("No state history entries yet.")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("persona-state-editor-toggle-button"))
    expect(screen.queryByTestId("persona-state-soul-input")).not.toBeInTheDocument()
    expect(screen.getByText("Show editor")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("persona-state-editor-toggle-button"))
    expect(screen.getByTestId("persona-state-soul-input")).toBeInTheDocument()
    expect(screen.getByText("Hide editor")).toBeInTheDocument()
  })
})
