import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
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
  useMobile: () => mediaMocks.isMobile
}))

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

vi.mock("@/services/acp/client", () => ({
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
}))

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
