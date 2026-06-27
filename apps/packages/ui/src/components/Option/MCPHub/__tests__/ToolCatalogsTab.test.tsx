// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

const mocks = vi.hoisted(() => ({
  getToolRegistrySummary: vi.fn(),
  getMcpHubReadiness: vi.fn(),
  listExternalServers: vi.fn(),
  refreshExternalServerDiscovery: vi.fn()
}))

vi.mock("@/services/tldw/mcp-hub", () => ({
  getToolRegistrySummary: (...args: unknown[]) => mocks.getToolRegistrySummary(...args),
  getMcpHubReadiness: (...args: unknown[]) => mocks.getMcpHubReadiness(...args),
  listExternalServers: (...args: unknown[]) => mocks.listExternalServers(...args),
  refreshExternalServerDiscovery: (...args: unknown[]) => mocks.refreshExternalServerDiscovery(...args)
}))

import { ToolCatalogsTab } from "../ToolCatalogsTab"

const readinessResponse = (overrides: Record<string, unknown> = {}) => ({
  display_state: "needs_setup",
  reason_codes: ["not_configured"],
  primary_reason_code: "not_configured",
  allowed_actions: ["add_server"],
  message: "Add an MCP server to begin setup.",
  servers: [],
  total_servers: 0,
  ready_server_count: 0,
  checking_server_count: 0,
  attention_server_count: 0,
  no_tool_server_count: 0,
  stale_server_count: 0,
  ...overrides
})

const readinessServer = (overrides: Record<string, unknown> = {}) => ({
  server_id: "docs-managed",
  server_name: "Docs Managed",
  display_state: "needs_attention",
  credential_state: "not_required",
  tool_count: 0,
  reason_codes: ["discovery_not_run"],
  primary_reason_code: "discovery_not_run",
  allowed_actions: ["refresh_discovery", "edit_config"],
  message: "Run discovery to populate this server's tool catalog. No credentials required.",
  current_operation: null,
  last_validation_at: null,
  last_discovery_at: null,
  last_successful_discovery_at: null,
  last_error_category: null,
  last_error_message: null,
  refresh_result: null,
  ...overrides
})

const externalServer = (overrides: Record<string, unknown> = {}) => ({
  id: "docs-managed",
  name: "Docs Managed",
  enabled: true,
  owner_scope_type: "global",
  transport: "stdio",
  config: {},
  secret_configured: false,
  server_source: "managed",
  binding_count: 0,
  runtime_executable: true,
  auth_template_present: false,
  auth_template_valid: false,
  auth_template_blocked_reason: "no_auth_template",
  credential_slots: [],
  ...overrides
})

describe("ToolCatalogsTab", () => {
  beforeEach(() => {
    vi.clearAllMocks()
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
    mocks.listExternalServers.mockResolvedValue([])
    mocks.getMcpHubReadiness.mockResolvedValue(readinessResponse())
    mocks.refreshExternalServerDiscovery.mockResolvedValue({})
  })

  it("renders registry-backed module and tool metadata", async () => {
    render(<ToolCatalogsTab />)

    expect(await screen.findByText("notes")).toBeTruthy()
    expect(screen.getByText("notes.search")).toBeTruthy()
    expect(screen.getByText("filesystem.read")).toBeTruthy()
    expect(screen.getByText("path-enforceable")).toBeTruthy()
    expect(screen.getByText("hints:path")).toBeTruthy()
  })

  it("shows add-server recovery when no MCP servers are connected", async () => {
    const user = userEvent.setup()
    const onOpenServerSetup = vi.fn()
    mocks.getToolRegistrySummary.mockResolvedValueOnce({
      entries: [],
      modules: []
    })

    render(<ToolCatalogsTab onOpenServerSetup={onOpenServerSetup} />)

    expect(await screen.findByText(/no mcp servers connected/i)).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /add server/i }))
    expect(onOpenServerSetup).toHaveBeenCalledTimes(1)
  })

  it("shows refresh discovery recovery when a saved server has no catalog yet", async () => {
    const user = userEvent.setup()
    mocks.getToolRegistrySummary.mockResolvedValueOnce({
      entries: [],
      modules: []
    })
    mocks.listExternalServers.mockResolvedValueOnce([externalServer()])
    mocks.getMcpHubReadiness.mockResolvedValueOnce(
      readinessResponse({
        display_state: "needs_attention",
        reason_codes: ["discovery_not_run"],
        primary_reason_code: "discovery_not_run",
        allowed_actions: ["refresh_discovery"],
        message: "Run discovery to populate MCP tool catalogs.",
        servers: [readinessServer()],
        total_servers: 1,
        attention_server_count: 1
      })
    )

    render(<ToolCatalogsTab />)

    expect(await screen.findByText(/docs managed is saved/i)).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /refresh discovery/i }))
    expect(mocks.refreshExternalServerDiscovery).toHaveBeenCalledWith("docs-managed")
  })

  it("shows credential recovery when a saved server is missing required credentials", async () => {
    const user = userEvent.setup()
    const onOpenServerCredentials = vi.fn()
    mocks.getToolRegistrySummary.mockResolvedValueOnce({
      entries: [],
      modules: []
    })
    mocks.listExternalServers.mockResolvedValueOnce([
      externalServer({
        credential_slots: [
          {
            server_id: "docs-managed",
            slot_name: "token_readonly",
            display_name: "Read-only token",
            secret_kind: "bearer_token",
            privilege_class: "read",
            is_required: true,
            secret_configured: false
          }
        ]
      })
    ])
    mocks.getMcpHubReadiness.mockResolvedValueOnce(
      readinessResponse({
        display_state: "needs_attention",
        reason_codes: ["auth_missing"],
        primary_reason_code: "auth_missing",
        allowed_actions: ["open_credentials", "view_details"],
        message: "One or more MCP servers are missing required credentials.",
        servers: [
          readinessServer({
            credential_state: "required_missing",
            reason_codes: ["auth_missing"],
            primary_reason_code: "auth_missing",
            allowed_actions: ["open_credentials", "view_details"],
            message: "Credentials are required before this server can be used."
          })
        ],
        total_servers: 1,
        attention_server_count: 1
      })
    )

    render(<ToolCatalogsTab onOpenServerCredentials={onOpenServerCredentials} />)

    expect(await screen.findByText(/docs managed needs credentials/i)).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /fix credentials/i }))
    expect(onOpenServerCredentials).toHaveBeenCalledWith("docs-managed")
  })

  it("shows server-config recovery when a saved server runtime is unavailable", async () => {
    const user = userEvent.setup()
    const onOpenServerConfig = vi.fn()
    mocks.getToolRegistrySummary.mockResolvedValueOnce({
      entries: [],
      modules: []
    })
    mocks.listExternalServers.mockResolvedValueOnce([
      externalServer({
        runtime_executable: false
      })
    ])
    mocks.getMcpHubReadiness.mockResolvedValueOnce(
      readinessResponse({
        display_state: "needs_attention",
        reason_codes: ["runtime_unavailable"],
        primary_reason_code: "runtime_unavailable",
        allowed_actions: ["edit_config", "view_details"],
        message: "One or more MCP server runtimes are unavailable.",
        servers: [
          readinessServer({
            reason_codes: ["runtime_unavailable"],
            primary_reason_code: "runtime_unavailable",
            allowed_actions: ["edit_config", "view_details"],
            message: "The configured runtime is not available."
          })
        ],
        total_servers: 1,
        attention_server_count: 1
      })
    )

    render(<ToolCatalogsTab onOpenServerConfig={onOpenServerConfig} />)

    expect(await screen.findByText(/docs managed runtime is unavailable/i)).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /open server config/i }))
    expect(onOpenServerConfig).toHaveBeenCalledWith("docs-managed")
  })

  it("shows preflight failure recovery with details", async () => {
    const user = userEvent.setup()
    const onOpenServerConfig = vi.fn()
    mocks.getToolRegistrySummary.mockResolvedValueOnce({
      entries: [],
      modules: []
    })
    mocks.listExternalServers.mockResolvedValueOnce([externalServer()])
    mocks.getMcpHubReadiness.mockResolvedValueOnce(
      readinessResponse({
        display_state: "needs_attention",
        reason_codes: ["preflight_failed"],
        primary_reason_code: "preflight_failed",
        allowed_actions: ["edit_config", "validate", "view_details"],
        message: "One or more MCP servers failed preflight validation.",
        servers: [
          readinessServer({
            reason_codes: ["preflight_failed"],
            primary_reason_code: "preflight_failed",
            allowed_actions: ["edit_config", "validate", "view_details"],
            message: "Preflight validation failed."
          })
        ],
        total_servers: 1,
        attention_server_count: 1
      })
    )

    render(<ToolCatalogsTab onOpenServerConfig={onOpenServerConfig} />)

    expect(await screen.findByText(/docs managed failed preflight validation/i)).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /open server config/i }))
    expect(onOpenServerConfig).toHaveBeenCalledWith("docs-managed")

    await user.click(screen.getByRole("button", { name: /view details/i }))
    expect(await screen.findByText(/reason codes: preflight_failed/i)).toBeTruthy()
  })

  it("shows unreachable recovery with config refresh and details actions", async () => {
    const user = userEvent.setup()
    const onOpenServerConfig = vi.fn()
    mocks.getToolRegistrySummary.mockResolvedValueOnce({
      entries: [],
      modules: []
    })
    mocks.listExternalServers.mockResolvedValueOnce([externalServer()])
    mocks.getMcpHubReadiness.mockResolvedValueOnce(
      readinessResponse({
        display_state: "needs_attention",
        reason_codes: ["unreachable"],
        primary_reason_code: "unreachable",
        allowed_actions: ["edit_config", "refresh_discovery", "view_details"],
        message: "One or more MCP servers are unreachable.",
        servers: [
          readinessServer({
            reason_codes: ["unreachable"],
            primary_reason_code: "unreachable",
            allowed_actions: ["edit_config", "refresh_discovery", "view_details"],
            message: "The server is unreachable."
          })
        ],
        total_servers: 1,
        attention_server_count: 1
      })
    )

    render(<ToolCatalogsTab onOpenServerConfig={onOpenServerConfig} />)

    expect(await screen.findByText(/docs managed is unreachable/i)).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /open server config/i }))
    expect(onOpenServerConfig).toHaveBeenCalledWith("docs-managed")

    await user.click(screen.getByRole("button", { name: /view details/i }))
    expect(await screen.findByText(/reason codes: unreachable/i)).toBeTruthy()

    await user.click(screen.getByRole("button", { name: /refresh discovery/i }))
    expect(mocks.refreshExternalServerDiscovery).toHaveBeenCalledWith("docs-managed")
  })

  it("shows discovery failure recovery with refresh and details actions", async () => {
    const user = userEvent.setup()
    mocks.getToolRegistrySummary.mockResolvedValueOnce({
      entries: [],
      modules: []
    })
    mocks.listExternalServers.mockResolvedValueOnce([externalServer()])
    mocks.getMcpHubReadiness.mockResolvedValueOnce(
      readinessResponse({
        display_state: "needs_attention",
        reason_codes: ["discovery_failed"],
        primary_reason_code: "discovery_failed",
        allowed_actions: ["refresh_discovery", "view_details"],
        message: "One or more MCP servers failed tool discovery.",
        servers: [
          readinessServer({
            reason_codes: ["discovery_failed"],
            primary_reason_code: "discovery_failed",
            allowed_actions: ["refresh_discovery", "view_details"],
            message: "Tool discovery failed.",
            last_error_message: "Process exited with code 1"
          })
        ],
        total_servers: 1,
        attention_server_count: 1
      })
    )

    render(<ToolCatalogsTab />)

    expect(await screen.findByText(/docs managed discovery failed/i)).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /view details/i }))
    expect(await screen.findByText(/process exited with code 1/i)).toBeTruthy()

    await user.click(screen.getByRole("button", { name: /refresh discovery/i }))
    expect(mocks.refreshExternalServerDiscovery).toHaveBeenCalledWith("docs-managed")
    expect(screen.queryByRole("button", { name: /open server config/i })).toBeNull()
  })

  it("shows in-progress discovery without a duplicate refresh action", async () => {
    const user = userEvent.setup()
    mocks.getToolRegistrySummary.mockResolvedValueOnce({
      entries: [],
      modules: []
    })
    mocks.listExternalServers.mockResolvedValueOnce([externalServer()])
    mocks.getMcpHubReadiness.mockResolvedValueOnce(
      readinessResponse({
        display_state: "checking",
        reason_codes: ["discovery_not_run"],
        primary_reason_code: "discovery_not_run",
        allowed_actions: ["view_details"],
        message: "Discovery is running.",
        servers: [
          readinessServer({
            display_state: "checking",
            reason_codes: ["discovery_not_run"],
            primary_reason_code: "discovery_not_run",
            allowed_actions: ["view_details"],
            message: "Discovery is running.",
            current_operation: {
              operation_type: "discovery",
              started_at: "2026-06-27T08:00:00Z",
              message: "Discovery is running."
            }
          })
        ],
        total_servers: 1,
        checking_server_count: 1
      })
    )

    render(<ToolCatalogsTab />)

    expect(await screen.findByText(/docs managed discovery is running/i)).toBeTruthy()
    expect(screen.queryByRole("button", { name: /refresh discovery/i })).toBeNull()

    await user.click(screen.getByRole("button", { name: /view details/i }))
    expect(await screen.findByText(/reason codes: discovery_not_run/i)).toBeTruthy()
  })

  it("keeps the catalog visible while surfacing partial capability warnings", async () => {
    mocks.listExternalServers.mockResolvedValueOnce([externalServer()])
    mocks.getMcpHubReadiness.mockResolvedValueOnce(
      readinessResponse({
        display_state: "ready",
        reason_codes: ["partial_capability"],
        primary_reason_code: "partial_capability",
        allowed_actions: ["open_tool_catalog", "view_details"],
        message: "Server is ready, but some capabilities need review.",
        servers: [
          readinessServer({
            display_state: "ready",
            tool_count: 1,
            reason_codes: ["partial_capability"],
            primary_reason_code: "partial_capability",
            allowed_actions: ["open_tool_catalog", "view_details"],
            message: "Server is ready, but some capabilities need review."
          })
        ],
        total_servers: 1,
        ready_server_count: 1
      })
    )

    render(<ToolCatalogsTab />)

    expect(await screen.findByText("notes.search")).toBeTruthy()
    expect(screen.getByText(/some capabilities need review/i)).toBeTruthy()
  })

  it("keeps the catalog visible when readiness metadata cannot be loaded", async () => {
    mocks.listExternalServers.mockResolvedValueOnce([externalServer()])
    mocks.getMcpHubReadiness.mockRejectedValueOnce(new Error("readiness offline"))

    render(<ToolCatalogsTab />)

    expect(await screen.findByText("notes.search")).toBeTruthy()
    expect(screen.getAllByText(/catalog recovery details are limited/i).length).toBeGreaterThan(0)
  })

  it("shows stale catalog recovery when server config changed", async () => {
    const user = userEvent.setup()
    mocks.getToolRegistrySummary.mockResolvedValueOnce({
      entries: [],
      modules: []
    })
    mocks.listExternalServers.mockResolvedValueOnce([externalServer()])
    mocks.getMcpHubReadiness.mockResolvedValueOnce(
      readinessResponse({
        display_state: "stale",
        reason_codes: ["config_changed"],
        primary_reason_code: "config_changed",
        allowed_actions: ["refresh_discovery", "edit_config"],
        message: "One or more MCP server configurations changed after discovery.",
        servers: [
          readinessServer({
            display_state: "stale",
            reason_codes: ["config_changed"],
            primary_reason_code: "config_changed",
            allowed_actions: ["refresh_discovery", "edit_config"],
            message: "Configuration changed after the last discovery run."
          })
        ],
        total_servers: 1,
        stale_server_count: 1
      })
    )

    render(<ToolCatalogsTab />)

    expect(await screen.findByText(/docs managed catalog is stale/i)).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /refresh discovery/i }))
    expect(mocks.refreshExternalServerDiscovery).toHaveBeenCalledWith("docs-managed")
  })
})
