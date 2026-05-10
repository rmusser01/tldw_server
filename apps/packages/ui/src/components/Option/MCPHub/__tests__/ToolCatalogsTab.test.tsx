// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

const mocks = vi.hoisted(() => ({
  getToolRegistrySummary: vi.fn(),
  invalidateQueries: vi.fn(),
  listExternalServers: vi.fn(),
  useMcpTools: vi.fn(),
  refreshExternalServerDiscovery: vi.fn()
}))

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({
    invalidateQueries: mocks.invalidateQueries
  })
}))

vi.mock("@/services/tldw/mcp-hub", () => ({
  describeExternalServerDiscoveryRefreshFailure: (result: { message?: string | null; errors?: Record<string, string> }) => {
    const errorText = Object.entries(result.errors ?? {})
      .map(([serverId, reason]) => `${serverId}: ${reason}`)
      .join("; ")
    return [result.message, errorText].filter(Boolean).join(" - ") || "Discovery refresh failed"
  },
  getToolRegistrySummary: (...args: unknown[]) => mocks.getToolRegistrySummary(...args),
  listExternalServers: (...args: unknown[]) => mocks.listExternalServers(...args),
  refreshExternalServerDiscovery: (...args: unknown[]) => mocks.refreshExternalServerDiscovery(...args)
}))

vi.mock("@/hooks/useMcpTools", () => ({
  useMcpTools: (...args: unknown[]) => mocks.useMcpTools(...args)
}))

import { ToolCatalogsTab } from "../ToolCatalogsTab"

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve
    reject = promiseReject
  })
  return { promise, resolve, reject }
}

const registrySummary = (toolName: string, moduleName: string) => ({
  entries: [
    {
      tool_name: toolName,
      display_name: toolName,
      module: moduleName,
      category: "search",
      risk_class: "low",
      capabilities: ["network.read"],
      mutates_state: false,
      uses_filesystem: false,
      uses_processes: false,
      uses_network: true,
      uses_credentials: false,
      supports_arguments_preview: true,
      path_boundable: false,
      path_argument_hints: [],
      metadata_source: "explicit",
      metadata_warnings: []
    }
  ],
  modules: [
    {
      module: moduleName,
      display_name: moduleName,
      tool_count: 1,
      risk_summary: { low: 1, medium: 0, high: 0, unclassified: 0 },
      metadata_warnings: []
    }
  ]
})

describe("ToolCatalogsTab", () => {
  beforeEach(() => {
    vi.resetAllMocks()
    mocks.invalidateQueries.mockResolvedValue(undefined)
    mocks.getToolRegistrySummary.mockResolvedValue({
      entries: [
        {
          tool_name: "notes.search",
          display_name: "notes.search",
          module: "notes",
          category: "search",
          risk_class: "low",
          capabilities: ["filesystem.read"],
          mutates_state: false,
          uses_filesystem: true,
          uses_processes: false,
          uses_network: false,
          uses_credentials: false,
          supports_arguments_preview: true,
          path_boundable: true,
          path_argument_hints: ["path"],
          metadata_source: "explicit",
          metadata_warnings: []
        }
      ],
      modules: [
        {
          module: "notes",
          display_name: "notes",
          tool_count: 1,
          risk_summary: { low: 1, medium: 0, high: 0, unclassified: 0 },
          metadata_warnings: []
        }
      ]
    })
    mocks.refreshExternalServerDiscovery.mockResolvedValue({
      ok: true,
      status: "refreshed"
    })
    mocks.listExternalServers.mockResolvedValue([
      {
        id: "docs-managed",
        name: "Docs Managed",
        enabled: true,
        owner_scope_type: "global",
        transport: "stdio",
        config: {},
        secret_configured: false,
        server_source: "managed",
        binding_count: 1,
        runtime_executable: true,
        auth_template_present: false,
        auth_template_valid: false,
        auth_template_blocked_reason: null,
        credential_slots: []
      }
    ])
    mocks.useMcpTools.mockReturnValue({
      healthState: "healthy",
      toolsAvailable: true,
      availableTools: [{ rawName: "notes.search" }],
      chatTools: [{ rawName: "notes.search" }],
      toolCounts: {
        discovered: 1,
        executable: 1,
        disabled: 0,
        colliding: 0,
        chatEnabled: 1
      }
    })
  })

  it("renders registry-backed module and tool metadata", async () => {
    render(<ToolCatalogsTab />)

    expect(await screen.findByText("notes")).toBeTruthy()
    expect(screen.getByText("notes.search")).toBeTruthy()
    expect(screen.getByText("filesystem.read")).toBeTruthy()
    expect(screen.getByText("path-enforceable")).toBeTruthy()
    expect(screen.getByText("hints:path")).toBeTruthy()
  })

  it("explicitly refreshes external discovery and reloads registry metadata", async () => {
    const user = userEvent.setup()
    mocks.getToolRegistrySummary
      .mockResolvedValueOnce({
        entries: [],
        modules: []
      })
      .mockResolvedValueOnce({
        entries: [
          {
            tool_name: "web.search",
            display_name: "web.search",
            module: "external:web",
            category: "search",
            risk_class: "medium",
            capabilities: ["network.read"],
            mutates_state: false,
            uses_filesystem: false,
            uses_processes: false,
            uses_network: true,
            uses_credentials: true,
            supports_arguments_preview: true,
            path_boundable: false,
            path_argument_hints: [],
            metadata_source: "explicit",
            metadata_warnings: []
          }
        ],
        modules: [
          {
            module: "external:web",
            display_name: "External Web",
            tool_count: 1,
            risk_summary: { low: 0, medium: 1, high: 0, unclassified: 0 },
            metadata_warnings: []
          }
        ]
      })

    render(<ToolCatalogsTab />)

    expect(await screen.findByText(/no tools discovered yet/i)).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /refresh tools/i }))

    expect(mocks.refreshExternalServerDiscovery).toHaveBeenCalledWith()
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-tools"] })
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-tool-catalogs"] })
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-tool-modules"] })
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-health"] })
    expect(await screen.findByText("External Web")).toBeTruthy()
    expect(screen.getByText("web.search")).toBeTruthy()
    expect(mocks.getToolRegistrySummary).toHaveBeenCalledTimes(2)
  })

  it("offers Add Server when no managed servers or tools exist", async () => {
    const user = userEvent.setup()
    const onAddServer = vi.fn()
    mocks.getToolRegistrySummary.mockResolvedValueOnce({
      entries: [],
      modules: []
    })
    mocks.listExternalServers.mockResolvedValueOnce([])
    mocks.useMcpTools.mockReturnValue({
      healthState: "healthy",
      toolsAvailable: false,
      availableTools: [],
      chatTools: [],
      toolCounts: {
        discovered: 0,
        executable: 0,
        disabled: 0,
        colliding: 0,
        chatEnabled: 0
      }
    })

    render(<ToolCatalogsTab onAddServer={onAddServer} />)

    expect(await screen.findByText(/no managed mcp servers yet/i)).toBeTruthy()
    expect(screen.getByText(/add a managed server before looking for tool catalog entries/i)).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /add server/i }))
    expect(onAddServer).toHaveBeenCalledTimes(1)
  })

  it("surfaces server inventory load errors instead of showing an empty server state", async () => {
    const user = userEvent.setup()
    mocks.getToolRegistrySummary
      .mockResolvedValueOnce({
        entries: [],
        modules: []
      })
      .mockResolvedValueOnce({
        entries: [],
        modules: []
      })
    mocks.listExternalServers
      .mockRejectedValueOnce(new Error("inventory timeout"))
      .mockResolvedValueOnce([])

    render(<ToolCatalogsTab />)

    expect(await screen.findByText(/could not load server inventory/i)).toBeTruthy()
    expect(screen.getByText(/inventory timeout/i)).toBeTruthy()
    expect(screen.getByText(/server inventory unavailable/i)).toBeTruthy()
    expect(screen.queryByText(/no managed mcp servers yet/i)).toBeNull()

    await user.click(screen.getByRole("button", { name: /retry server inventory/i }))

    expect(await screen.findByText(/no managed mcp servers yet/i)).toBeTruthy()
    expect(screen.queryByText(/server inventory unavailable/i)).toBeNull()
  })

  it("offers discovery refresh when managed servers exist but no tools are registered", async () => {
    const user = userEvent.setup()
    mocks.getToolRegistrySummary
      .mockResolvedValueOnce({
        entries: [],
        modules: []
      })
      .mockResolvedValueOnce(registrySummary("docs.lookup", "External Docs"))

    render(<ToolCatalogsTab />)

    expect(await screen.findByText(/no tools discovered yet/i)).toBeTruthy()
    expect(screen.getByText(/managed servers are configured/i)).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /refresh discovery/i }))

    expect(mocks.refreshExternalServerDiscovery).toHaveBeenCalledWith()
    expect(await screen.findByText("docs.lookup")).toBeTruthy()
  })

  it("shows access guidance when registry tools are present but none are executable in chat", async () => {
    mocks.useMcpTools.mockReturnValue({
      healthState: "healthy",
      toolsAvailable: true,
      availableTools: [{ rawName: "notes.search" }],
      chatTools: [],
      toolCounts: {
        discovered: 1,
        executable: 1,
        disabled: 0,
        colliding: 0,
        chatEnabled: 0
      }
    })

    render(<ToolCatalogsTab />)

    expect(await screen.findByText("notes.search")).toBeTruthy()
    expect(screen.getByText(/tools are registered but not executable in chat/i)).toBeTruthy()
    expect(screen.getByText(/review profile assignments and disabled tool settings/i)).toBeTruthy()
  })

  it("surfaces explicit refresh failures without clearing the current registry", async () => {
    const user = userEvent.setup()
    mocks.refreshExternalServerDiscovery.mockRejectedValueOnce(new Error("runtime unavailable"))

    render(<ToolCatalogsTab />)

    expect(await screen.findByText("notes.search")).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /refresh tools/i }))

    expect(await screen.findByText(/failed to refresh tool discovery/i)).toBeTruthy()
    expect(screen.getByText(/runtime unavailable/i)).toBeTruthy()
    expect(screen.getByText("notes.search")).toBeTruthy()
    expect(mocks.getToolRegistrySummary).toHaveBeenCalledTimes(1)
  })

  it("surfaces resolved refresh errors without clearing the current registry", async () => {
    const user = userEvent.setup()
    mocks.refreshExternalServerDiscovery.mockResolvedValueOnce({
      ok: false,
      message: "Discovery refreshed with errors",
      errors: { docs: "external_server_discovery_failed" }
    })

    render(<ToolCatalogsTab />)

    expect(await screen.findByText("notes.search")).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /refresh tools/i }))

    expect(await screen.findByText(/failed to refresh tool discovery/i)).toBeTruthy()
    expect(screen.getByText(/external_server_discovery_failed/i)).toBeTruthy()
    expect(screen.getByText("notes.search")).toBeTruthy()
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-tools"] })
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-tool-catalogs"] })
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-tool-modules"] })
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-health"] })
    expect(mocks.getToolRegistrySummary).toHaveBeenCalledTimes(2)
  })

  it("ignores stale registry loads that resolve after a newer refresh", async () => {
    const user = userEvent.setup()
    const stale = deferred<ReturnType<typeof registrySummary>>()
    const fresh = deferred<ReturnType<typeof registrySummary>>()
    mocks.getToolRegistrySummary
      .mockReturnValueOnce(stale.promise)
      .mockReturnValueOnce(fresh.promise)

    render(<ToolCatalogsTab />)

    await user.click(screen.getByRole("button", { name: /refresh tools/i }))
    await waitFor(() => {
      expect(mocks.getToolRegistrySummary).toHaveBeenCalledTimes(2)
    })

    await act(async () => {
      fresh.resolve(registrySummary("web.search", "External Web"))
    })
    expect(await screen.findByText("web.search")).toBeTruthy()

    await act(async () => {
      stale.resolve(registrySummary("stale.search", "Stale Module"))
    })
    await waitFor(() => {
      expect(screen.getByText("web.search")).toBeTruthy()
    })
    expect(screen.queryByText("stale.search")).toBeNull()
  })
})
