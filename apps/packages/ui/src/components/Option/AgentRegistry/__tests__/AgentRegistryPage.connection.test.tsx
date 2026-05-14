import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AgentRegistryPage } from "../index"

const baseACPHealthPayload = {
  runner: "ok",
  agent: "ok",
  api_keys: "ok"
}

const baseExecutionHealthSummaryPayload = {
  timestamp: "2026-05-14T09:00:00Z",
  range_days: 30,
  sessions: {
    total: 0,
    by_status: {}
  },
  failure_buckets: {
    setup_blockers: 0,
    runner_session_failures: 0,
    reviewer_rejections: 0,
    reviewer_failures: 0,
    governance_denials: 0,
    structured_completion_failures: 0,
    sandbox_runtime_errors: 0,
    retention_redaction_actions: 0
  },
  setup_health: {
    agent: { status: "unknown", blockers: [], evidence_count: 0 },
    workspace: { status: "unknown", blockers: [], evidence_count: 0 },
    sandbox_runtime: { status: "unknown", blockers: [], evidence_count: 0 },
    mcp_injection: { status: "unknown", blockers: [], evidence_count: 0 },
    scheduler_trigger_path: { status: "unknown", blockers: [], evidence_count: 0 }
  },
  agents: [],
  compatibility: {
    by_support_state: {},
    documented_unverified_agents: [],
    live_certification_required: false,
    docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md"
  },
  retention: {
    session_retention_days: 30,
    audit_retention_days: 30,
    policy: "closed_error_sessions_and_audit_events_purged_after_retention"
  },
  redaction: {
    detail_events_artifacts_redacted_views: true,
    diagnostics_sanitized: true,
    audit_metadata_sanitized: true
  }
}

const installACPFetchMock = ({
  healthPayload = baseACPHealthPayload,
  summaryPayload = baseExecutionHealthSummaryPayload,
  summaryOk = true,
  summaryStatus = summaryOk ? 200 : 403
}: {
  healthPayload?: Record<string, unknown>
  summaryPayload?: Record<string, unknown>
  summaryOk?: boolean
  summaryStatus?: number
} = {}) => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes("/api/v1/admin/acp/execution-health/summary")) {
        return {
          ok: summaryOk,
          status: summaryStatus,
          json: async () => summaryPayload
        }
      }
      return {
        ok: true,
        status: 200,
        json: async () => healthPayload
      }
    })
  )
}

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
  getAvailableAgents: vi.fn()
}))

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()
  const labelOverrides: Partial<
    Record<Parameters<typeof actual.getDesignSystemState>[0], string>
  > = {
    ready: "Registry ready",
    setup_required: "Registry setup required"
  }

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)
        const label = labelOverrides[key]
        return label ? { ...state, label } : state
      }
    )
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | Record<string, unknown>,
      options?: Record<string, unknown>
    ) => {
      const values =
        typeof fallbackOrOptions === "object" && fallbackOrOptions !== null
          ? fallbackOrOptions
          : options
      const template =
        typeof fallbackOrOptions === "string"
          ? fallbackOrOptions
          : typeof fallbackOrOptions?.defaultValue === "string"
            ? String(fallbackOrOptions.defaultValue)
            : key

      return template.replace(/\{\{(\w+)}}/g, (_, token: string) =>
        values?.[token] == null ? "" : String(values[token])
      )
    }
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

    async getAvailableAgents() {
      return acpMocks.getAvailableAgents(this.config)
    }
  }
}))

vi.mock("antd", () => ({
  Alert: ({
    message,
    description
  }: {
    message?: React.ReactNode
    description?: React.ReactNode
  }) => (
    <div>
      <div>{message}</div>
      {description ? <div>{description}</div> : null}
    </div>
  ),
  Badge: ({ count }: { count?: React.ReactNode }) => <span>{count}</span>,
  Button: ({
    children,
    onClick
  }: {
    children?: React.ReactNode
    onClick?: () => void
  }) => (
    <button type="button" onClick={onClick}>
      {children}
    </button>
  ),
  Card: ({
    title,
    extra,
    children
  }: {
    title?: React.ReactNode
    extra?: React.ReactNode
    children?: React.ReactNode
  }) => (
    <section>
      {title}
      {extra}
      {children}
    </section>
  ),
  Empty: ({ description }: { description?: React.ReactNode }) => (
    <div>{description}</div>
  ),
  Spin: () => <div>Loading...</div>,
  Tag: ({ children }: { children?: React.ReactNode }) => <span>{children}</span>,
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>
}))

describe("AgentRegistryPage connection config", () => {
  const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE

  beforeEach(() => {
    vi.clearAllMocks()
    acpMocks.constructedConfigs.length = 0

    storageMocks.useStorage.mockImplementation((key: string, fallback: string) => {
      if (key === "serverUrl") return ["http://localhost:8000", vi.fn()]
      if (key === "authMode") return ["single-user", vi.fn()]
      if (key === "apiKey") return ["", vi.fn()]
      if (key === "accessToken") return ["", vi.fn()]
      return [fallback, vi.fn()]
    })

    configMocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "real-key",
      accessToken: ""
    })

    acpMocks.getAvailableAgents.mockResolvedValue({
      agents: [
        {
          type: "planner",
          name: "Planner Agent",
          description: "Plans work",
          is_configured: true
        }
      ]
    })

    installACPFetchMock()
  })

  afterEach(() => {
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }
  })

  it("uses the canonical tldw config for ACP requests instead of stale legacy storage keys", async () => {
    render(<AgentRegistryPage />)

    expect(await screen.findByText("Planner Agent")).toBeInTheDocument()

    await waitFor(async () => {
      expect(acpMocks.constructedConfigs[0]?.serverUrl).toBe("http://127.0.0.1:8000")
      expect(global.fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/acp/health",
        expect.objectContaining({
          headers: expect.objectContaining({
            "X-API-KEY": "real-key"
          })
        })
      )
      expect(global.fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/admin/acp/execution-health/summary?range_days=30",
        expect.objectContaining({
          headers: expect.objectContaining({
            "X-API-KEY": "real-key"
          })
        })
      )
      const authHeaders = await acpMocks.constructedConfigs[0]?.getAuthHeaders()
      expect(authHeaders).toEqual(
        expect.objectContaining({
          "X-API-KEY": "real-key"
        })
      )
    })
  })

  it("uses the shared quickstart health path instead of concatenating the backend serverUrl", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"

    render(<AgentRegistryPage />)

    expect(await screen.findByText("Planner Agent")).toBeInTheDocument()

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "/api/v1/acp/health",
        expect.objectContaining({
          headers: expect.objectContaining({
            "X-API-KEY": "real-key"
          })
        })
      )
      expect(global.fetch).toHaveBeenCalledWith(
        "/api/v1/admin/acp/execution-health/summary?range_days=30",
        expect.objectContaining({
          headers: expect.objectContaining({
            "X-API-KEY": "real-key"
          })
        })
      )
    })
  })

  it("uses design-system state labels for runtime setup states", async () => {
    acpMocks.getAvailableAgents.mockResolvedValue({
      agents: [
        {
          type: "planner",
          name: "Planner Agent",
          description: "Plans work",
          is_configured: true
        },
        {
          type: "local-runner",
          name: "Local Runner",
          description: "Requires local setup",
          is_configured: false
        }
      ]
    })

    render(<AgentRegistryPage />)

    expect(await screen.findByText("Planner Agent")).toBeInTheDocument()
    expect(screen.getByText("Local Runner")).toBeInTheDocument()
    expect(screen.getByText("Registry ready")).toBeInTheDocument()
    expect(screen.getByText("Registry setup required")).toBeInTheDocument()
  })

  it("normalizes structured ACP health payloads without trying to render raw objects", async () => {
    installACPFetchMock({
      healthPayload: {
        runner: {
          status: "ok",
          path: "/opt/homebrew/bin/go",
          source: "PATH"
        },
        agents: [
          {
            agent_type: "planner",
            status: "available",
            api_key_set: true
          }
        ],
        overall: "ok",
        message: null
      }
    })

    render(<AgentRegistryPage />)

    expect(await screen.findByText(/Runner source PATH path \/opt\/homebrew\/bin\/go/i)).toBeInTheDocument()
    expect(screen.getByText(/1\/1 agents available/i)).toBeInTheDocument()
  })

  it("renders ACP compatibility support and verification states separately from runtime setup", async () => {
    acpMocks.getAvailableAgents.mockResolvedValue({
      agents: [
        {
          type: "codex",
          name: "Codex CLI",
          description: "Configured locally",
          is_configured: true,
          support_state: "documented_unverified",
          verification_level: "documented_only",
          compatibility_notes: "Configured locally, but live-agent compatibility is not certified.",
          compatibility_docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md"
        },
        {
          type: "stub",
          name: "In-repo ACP stub",
          description: "Protocol fixture",
          is_configured: true,
          support_state: "supported_with_caveats",
          verification_level: "stub_smoke_tested",
          compatibility_notes: "Stub protocol coverage only."
        }
      ]
    })

    render(<AgentRegistryPage />)

    expect(await screen.findByText("Codex CLI")).toBeInTheDocument()
    expect(screen.getByText("documented_unverified")).toBeInTheDocument()
    expect(screen.getByText("documented_only")).toBeInTheDocument()
    expect(screen.getByText("supported_with_caveats")).toBeInTheDocument()
    expect(screen.getByText("stub_smoke_tested")).toBeInTheDocument()
    expect(screen.getByText(/configured but compatibility is documented_unverified/i)).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /Compatibility matrix/i })).toHaveAttribute(
      "href",
      "/docs-static/Development/ACP_Compatibility_Matrix.md"
    )
  })

  it("renders the admin execution-health summary without overstating agent certification", async () => {
    installACPFetchMock({
      summaryPayload: {
        ...baseExecutionHealthSummaryPayload,
        range_days: 30,
        sessions: {
          total: 3,
          by_status: {
            active: 2,
            error: 1
          }
        },
        failure_buckets: {
          setup_blockers: 1,
          runner_session_failures: 1,
          reviewer_rejections: 1,
          reviewer_failures: 0,
          governance_denials: 1,
          structured_completion_failures: 1,
          sandbox_runtime_errors: 1,
          retention_redaction_actions: 1
        },
        setup_health: {
          agent: { status: "blocked", blockers: ["adapter_required"], evidence_count: 1 },
          workspace: { status: "unknown", blockers: [], evidence_count: 0 },
          sandbox_runtime: { status: "blocked", blockers: ["sandbox_runtime_error"], evidence_count: 1 },
          mcp_injection: { status: "unknown", blockers: [], evidence_count: 0 },
          scheduler_trigger_path: { status: "ok", blockers: [], evidence_count: 2 }
        },
        compatibility: {
          by_support_state: {
            documented_unverified: 1,
            supported_with_caveats: 1
          },
          documented_unverified_agents: ["codex"],
          live_certification_required: true,
          docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md"
        },
        retention: {
          session_retention_days: 30,
          audit_retention_days: 45,
          policy: "closed_error_sessions_and_audit_events_purged_after_retention"
        }
      }
    })

    render(<AgentRegistryPage />)

    expect(await screen.findByText("ACP Execution Health")).toBeInTheDocument()
    expect(await screen.findByText("3 sessions in 30d")).toBeInTheDocument()
    expect(screen.getByText("2 active")).toBeInTheDocument()
    expect(screen.getByText("1 error")).toBeInTheDocument()
    expect(screen.getByText("Runner/session failures")).toBeInTheDocument()
    expect(screen.getByText("Setup blockers")).toBeInTheDocument()
    expect(screen.getByText("Unverified agents: codex")).toBeInTheDocument()
    expect(screen.getByText("Live certification required")).toBeInTheDocument()
    expect(screen.getByText("Agent blocked: adapter_required")).toBeInTheDocument()
    expect(screen.getByText("Retention 30d sessions / 45d audit")).toBeInTheDocument()
    expect(screen.getByText("Redacted drill-through enabled")).toBeInTheDocument()
  })

  it("keeps the registry usable when the admin execution-health summary is unavailable", async () => {
    installACPFetchMock({ summaryOk: false, summaryStatus: 403 })

    render(<AgentRegistryPage />)

    expect(await screen.findByText("Planner Agent")).toBeInTheDocument()
    expect(screen.getByText("Execution health summary unavailable")).toBeInTheDocument()
    expect(screen.getByText("Runner Binary")).toBeInTheDocument()
  })

  it("treats malformed admin execution-health summaries as unavailable", async () => {
    installACPFetchMock({
      summaryPayload: {
        sessions: {
          total: 1
        }
      }
    })

    render(<AgentRegistryPage />)

    expect(await screen.findByText("Planner Agent")).toBeInTheDocument()
    expect(screen.getByText("Execution health summary unavailable")).toBeInTheDocument()
  })

  it("uses safe defaults for partial admin execution-health summaries", async () => {
    installACPFetchMock({
      summaryPayload: {
        ...baseExecutionHealthSummaryPayload,
        sessions: {
          total: 2
        },
        failure_buckets: null,
        setup_health: null,
        compatibility: null,
        retention: null,
        redaction: null
      }
    })

    render(<AgentRegistryPage />)

    expect(await screen.findByText("Planner Agent")).toBeInTheDocument()
    expect(await screen.findByText("2 sessions in 30d")).toBeInTheDocument()
    expect(screen.getByText("No recent failure buckets")).toBeInTheDocument()
    expect(screen.getByText("No setup blockers in this window")).toBeInTheDocument()
    expect(screen.getByText("Review redaction settings")).toBeInTheDocument()
  })
})
