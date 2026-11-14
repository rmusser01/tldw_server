import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ACPPlayground } from "../index"
import { useACPSessionsStore } from "@/store/acp-sessions"

const storageMocks = vi.hoisted(() => ({
  useStorage: vi.fn()
}))

const configMocks = vi.hoisted(() => ({
  getConfig: vi.fn()
}))

const acpMocks = vi.hoisted(() => ({
  sessionOptions: null as null | { onError: (message: { session_id: string; message: string; type: string; code: string }) => void },
  constructedConfigs: [] as Array<{
    serverUrl: string
    getAuthHeaders: () => Promise<Record<string, string>>
    getAuthParams: () => Promise<{ token?: string; api_key?: string }>
  }>,
  listSessions: vi.fn(),
  getSessionDetail: vi.fn(),
  getSessionUsage: vi.fn()
}))

const mediaMocks = vi.hoisted(() => ({
  isMobile: false
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (...args: unknown[]) => storageMocks.useStorage(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) => configMocks.getConfig(...args)
  }
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => mediaMocks.isMobile,
  useDesktop: () => !mediaMocks.isMobile
}))

vi.mock("@/hooks/useACPSession", () => ({
  useACPSession: (options: typeof acpMocks.sessionOptions) => {
    acpMocks.sessionOptions = options
    return ({
    state: "disconnected",
    isConnected: false,
    error: null,
    connect: vi.fn(),
    sendPrompt: vi.fn(),
    cancel: vi.fn(),
    approvePermission: vi.fn(),
    denyPermission: vi.fn()
  })
  }
}))

vi.mock("@/services/acp/client", async (importActual) => {
  const actual = await importActual<typeof import("@/services/acp/client")>()
  return {
    ...actual,
    ACPRestClient: class {
      private config: {
        serverUrl: string
        getAuthHeaders: () => Promise<Record<string, string>>
        getAuthParams: () => Promise<{ token?: string; api_key?: string }>
      }

      constructor(config: {
        serverUrl: string
        getAuthHeaders: () => Promise<Record<string, string>>
        getAuthParams: () => Promise<{ token?: string; api_key?: string }>
      }) {
        this.config = config
        acpMocks.constructedConfigs.push(config)
      }

      async listSessions(params: { limit: number; offset: number }) {
        return acpMocks.listSessions(this.config, params)
      }

      async getSessionDetail(sessionId: string) {
        return acpMocks.getSessionDetail(this.config, sessionId)
      }

      async getSessionUsage(sessionId: string) {
        return acpMocks.getSessionUsage(this.config, sessionId)
      }
    }
  }
})

vi.mock("antd", () => ({
  Drawer: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
  Tabs: ({
    activeKey,
    items
  }: {
    activeKey?: string
    items?: Array<{ children?: React.ReactNode }>
  }) => (
    <div data-testid={`acp-tabs-${activeKey ?? "none"}`}>
      {items?.map((item, index) => <div key={index}>{item.children}</div>)}
    </div>
  )
}))

vi.mock("../ACPPlaygroundHeader", () => ({
  ACPPlaygroundHeader: () => <div>Agent Playground</div>
}))

vi.mock("../ACPSessionPanel", () => ({
  ACPSessionPanel: () => <div>Sessions</div>
}))

vi.mock("../ACPChatPanel", () => ({
  ACPChatPanel: () => <div>Chat</div>
}))

vi.mock("../ACPToolsPanel", () => ({
  ACPToolsPanel: () => <div>Tools</div>
}))

vi.mock("../ACPPermissionModal", () => ({
  ACPPermissionModal: () => null
}))

vi.mock("../ACPWorkspacePanel", () => ({
  ACPWorkspacePanel: () => <div>Workspace</div>
}))

describe("ACPPlayground canonical connection config", () => {
  const renderPlayground = () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false
        }
      }
    })
    return render(
      <QueryClientProvider client={queryClient}>
        <ACPPlayground />
      </QueryClientProvider>
    )
  }

  beforeEach(() => {
    useACPSessionsStore.getState().reset()
    vi.clearAllMocks()
    mediaMocks.isMobile = false
    window.history.pushState({}, "", "/acp-playground")
    acpMocks.constructedConfigs.length = 0

    storageMocks.useStorage.mockImplementation((key: string, fallback: unknown) => {
      if (key === "acp-playground-left-pane") return [true, vi.fn(), { isLoading: false }]
      if (key === "acp-playground-right-pane") return [true, vi.fn(), { isLoading: false }]
      if (key === "serverUrl") return ["http://localhost:8000", vi.fn(), { isLoading: false }]
      if (key === "authMode") return ["single-user", vi.fn(), { isLoading: false }]
      if (key === "apiKey") return ["", vi.fn(), { isLoading: false }]
      if (key === "accessToken") return ["", vi.fn(), { isLoading: false }]
      return [fallback, vi.fn(), { isLoading: false }]
    })

    configMocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "real-key",
      accessToken: ""
    })

    acpMocks.listSessions.mockResolvedValue({ sessions: [] })
    acpMocks.getSessionDetail.mockResolvedValue(null)
    acpMocks.getSessionUsage.mockResolvedValue(null)
  })

  it("hydrates ACP sessions with the canonical web config instead of stale legacy storage values", async () => {
    renderPlayground()

    await waitFor(async () => {
      expect(acpMocks.constructedConfigs[0]?.serverUrl).toBe("http://127.0.0.1:8000")
      const authHeaders = await acpMocks.constructedConfigs[0]?.getAuthHeaders()
      expect(authHeaders).toEqual(
        expect.objectContaining({
          "X-API-KEY": "real-key"
        })
      )
      expect(acpMocks.listSessions).toHaveBeenCalledWith(acpMocks.constructedConfigs[0], {
        limit: 200,
        offset: 0
      })
    })
  })

  it.each([["prompt_failed", 0], ["prompt_busy", 1], ["permission_not_found", 1], ["cancel_failed", 1]] as const)("retires permissions only for terminal failure: %s", async (code, pendingCount) => {
    renderPlayground()
    await waitFor(() => expect(acpMocks.listSessions).toHaveBeenCalled())
    let id = ""
    act(() => {
      const store = useACPSessionsStore.getState()
      id = store.createSession({ cwd: "/tmp/c05-disposable" })
      store.setActiveSession(id)
    })
    await waitFor(() => expect(acpMocks.getSessionDetail).toHaveBeenCalled())
    act(() => {
      useACPSessionsStore.getState().addPendingPermission(id, {
        request_id: "timeout-write", tool_name: "fs/write_text_file",
        tool_arguments: { path: "/tmp/c05-disposable/output.txt" },
        tier: "individual", timeout_seconds: 300, requestedAt: new Date()
      })
    })
    expect(useACPSessionsStore.getState().getSession(id)?.pendingPermissions).toHaveLength(1)
    act(() => acpMocks.sessionOptions!.onError({
      type: "error", code, session_id: id, message: "Controlled tool response deadline"
    }))
    const store = useACPSessionsStore.getState()
    expect(store.getSession(id)?.state).toBe("error")
    expect(store.getSession(id)?.pendingPermissions).toHaveLength(pendingCount)
    expect(store.globalError).toBe("Controlled tool response deadline")
  })

  it("routes health through the shared cookie-aware ACP transport", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    configMocks.getConfig.mockResolvedValue({
      serverUrl: window.location.origin,
      authMode: "single-user",
      authSource: "cookie-session"
    })

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ overall: "ok" })
    })
    vi.stubGlobal("fetch", fetchMock)
    renderPlayground()

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/v1/acp/health",
        expect.objectContaining({ credentials: "same-origin" })
      )
    })
    const headers = new Headers(fetchMock.mock.calls[0][1].headers)
    expect(headers.get("X-API-KEY")).toBeNull()
    expect(headers.get("Authorization")).toBeNull()
    vi.unstubAllGlobals()
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  })

  it("activates ACP sessions and session views from history deep links", async () => {
    mediaMocks.isMobile = true
    window.history.pushState(
      {},
      "",
      "/acp-playground?session=sess-linked&view=diagnostics"
    )
    acpMocks.listSessions.mockResolvedValue({
      sessions: [
        {
          session_id: "sess-linked",
          name: "Linked Session",
          status: "active",
          updated_at: "2026-05-13T14:00:00.000Z"
        }
      ]
    })

    renderPlayground()

    await waitFor(() => {
      expect(useACPSessionsStore.getState().activeSessionId).toBe("sess-linked")
    })
    expect(screen.getByTestId("acp-tabs-sessions")).toBeInTheDocument()
  })
})
