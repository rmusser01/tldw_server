import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AgentRegistryPage } from "../index"

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

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          runner: "ok",
          agent: "ok",
          api_keys: "ok"
        })
      }))
    )
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
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
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
        })
      }))
    )

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
})
