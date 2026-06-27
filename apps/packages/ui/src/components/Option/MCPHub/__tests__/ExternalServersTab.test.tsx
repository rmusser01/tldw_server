// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Modal } from "antd"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

const mocks = vi.hoisted(() => ({
  invalidateQueries: vi.fn(),
  listExternalServers: vi.fn(),
  setExternalServerSecret: vi.fn(),
  setExternalServerSlotSecret: vi.fn(),
  clearExternalServerSlotSecret: vi.fn(),
  getExternalServerAuthTemplate: vi.fn(),
  getMcpHubReadiness: vi.fn(),
  getToolRegistrySummary: vi.fn(),
  refreshExternalServerDiscovery: vi.fn(),
  refreshExternalServerReadinessDiscovery: vi.fn(),
  validateExternalServer: vi.fn(),
  updateExternalServerAuthTemplate: vi.fn(),
  importExternalServer: vi.fn(),
  createExternalServer: vi.fn(),
  updateExternalServer: vi.fn(),
  deleteExternalServer: vi.fn(),
  createExternalServerCredentialSlot: vi.fn(),
  updateExternalServerCredentialSlot: vi.fn(),
  deleteExternalServerCredentialSlot: vi.fn()
}))

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({
    invalidateQueries: mocks.invalidateQueries
  })
}))

vi.mock("@/services/tldw/mcp-hub", () => ({
  listExternalServers: (...args: unknown[]) => mocks.listExternalServers(...args),
  describeExternalServerDiscoveryRefreshFailure: (result: { message?: string | null; errors?: Record<string, string> }) => {
    const errorText = Object.entries(result.errors ?? {})
      .map(([serverId, reason]) => `${serverId}: ${reason}`)
      .join("; ")
    return [result.message, errorText].filter(Boolean).join(" - ") || "Discovery refresh failed"
  },
  setExternalServerSecret: (...args: unknown[]) => mocks.setExternalServerSecret(...args),
  setExternalServerSlotSecret: (...args: unknown[]) => mocks.setExternalServerSlotSecret(...args),
  clearExternalServerSlotSecret: (...args: unknown[]) => mocks.clearExternalServerSlotSecret(...args),
  getExternalServerAuthTemplate: (...args: unknown[]) => mocks.getExternalServerAuthTemplate(...args),
  getMcpHubReadiness: (...args: unknown[]) => mocks.getMcpHubReadiness(...args),
  getToolRegistrySummary: (...args: unknown[]) => mocks.getToolRegistrySummary(...args),
  refreshExternalServerDiscovery: (...args: unknown[]) => mocks.refreshExternalServerDiscovery(...args),
  refreshExternalServerReadinessDiscovery: (...args: unknown[]) => mocks.refreshExternalServerReadinessDiscovery(...args),
  validateExternalServer: (...args: unknown[]) => mocks.validateExternalServer(...args),
  updateExternalServerAuthTemplate: (...args: unknown[]) => mocks.updateExternalServerAuthTemplate(...args),
  importExternalServer: (...args: unknown[]) => mocks.importExternalServer(...args),
  createExternalServer: (...args: unknown[]) => mocks.createExternalServer(...args),
  updateExternalServer: (...args: unknown[]) => mocks.updateExternalServer(...args),
  deleteExternalServer: (...args: unknown[]) => mocks.deleteExternalServer(...args),
  createExternalServerCredentialSlot: (...args: unknown[]) => mocks.createExternalServerCredentialSlot(...args),
  updateExternalServerCredentialSlot: (...args: unknown[]) => mocks.updateExternalServerCredentialSlot(...args),
  deleteExternalServerCredentialSlot: (...args: unknown[]) => mocks.deleteExternalServerCredentialSlot(...args)
}))

import { ExternalServersTab } from "../ExternalServersTab"

const toolEntry = (serverId: string, toolName = "search") => ({
  tool_name: `ext.${serverId}.${toolName}`,
  display_name: toolName,
  description: null,
  module: `external.${serverId}`,
  module_display_name: serverId,
  category: "external",
  risk_class: "low",
  capabilities: [],
  mutates_state: false,
  uses_filesystem: false,
  uses_processes: false,
  uses_network: false,
  uses_credentials: false,
  supports_arguments_preview: false,
  path_boundable: false,
  path_argument_hints: [],
  metadata_source: "explicit",
  metadata_warnings: []
})

const readinessServer = (overrides: Record<string, unknown>) => ({
  server_id: "docs-managed",
  server_name: "Docs Managed",
  display_state: "needs_attention",
  credential_state: "required_missing",
  tool_count: 0,
  reason_codes: ["auth_missing", "discovery_not_run"],
  primary_reason_code: "auth_missing",
  allowed_actions: ["open_credentials", "view_details"],
  message: "Credentials are required before this server can be used.",
  current_operation: null,
  last_validation_at: null,
  last_discovery_at: null,
  last_successful_discovery_at: null,
  last_error_category: null,
  last_error_message: null,
  refresh_result: null,
  ...overrides
})

const readinessResponse = (servers: Array<Record<string, unknown>>) => ({
  display_state: servers.some((server) => server.display_state === "stale")
    ? "stale"
    : servers.every((server) => server.display_state === "ready")
      ? "ready"
      : "needs_attention",
  reason_codes: servers.flatMap((server) => server.reason_codes as string[]),
  primary_reason_code: (servers[0]?.primary_reason_code as string | null | undefined) ?? null,
  allowed_actions: [],
  message: "MCP Hub readiness loaded.",
  servers,
  total_servers: servers.length,
  ready_server_count: servers.filter((server) => server.display_state === "ready").length,
  checking_server_count: 0,
  attention_server_count: servers.filter((server) => server.display_state === "needs_attention").length,
  no_tool_server_count: 0,
  stale_server_count: servers.filter((server) => server.display_state === "stale").length
})

const findServerRow = async (name: string) => {
  const matches = await screen.findAllByText(name)
  const row = matches
    .map((match) => match.closest(".ant-list-item"))
    .find((item): item is HTMLElement => Boolean(item))
  if (!row) {
    throw new Error(`Expected to find server row for ${name}`)
  }
  return row
}

describe("ExternalServersTab", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.invalidateQueries.mockResolvedValue(undefined)
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
        binding_count: 2,
        runtime_executable: true,
        auth_template_present: false,
        auth_template_valid: false,
        auth_template_blocked_reason: "no_auth_template",
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
      },
      {
        id: "search-legacy",
        name: "Search Legacy",
        enabled: true,
        owner_scope_type: "global",
        transport: "websocket",
        config: {},
        secret_configured: false,
        server_source: "legacy",
        binding_count: 0,
        runtime_executable: false,
        auth_template_present: false,
        auth_template_valid: false,
        auth_template_blocked_reason: "no_auth_template"
      },
      {
        id: "docs-legacy",
        name: "Docs Legacy",
        enabled: true,
        owner_scope_type: "global",
        transport: "stdio",
        config: {},
        secret_configured: false,
        server_source: "legacy",
        superseded_by_server_id: "docs-managed",
        binding_count: 0,
        runtime_executable: false,
        auth_template_present: false,
        auth_template_valid: false,
        auth_template_blocked_reason: "no_auth_template"
      }
    ])
    mocks.getExternalServerAuthTemplate.mockImplementation(async (serverId: string) => {
      if (serverId === "docs-managed") {
        return {
          mode: "template",
          mappings: []
        }
      }
      return {
        mode: "template",
        mappings: []
      }
    })
    mocks.getMcpHubReadiness.mockResolvedValue(
      readinessResponse([
        readinessServer({
          server_id: "docs-managed",
          server_name: "Docs Managed"
        })
      ])
    )
    mocks.getToolRegistrySummary.mockResolvedValue({
      entries: [],
      modules: []
    })
    mocks.validateExternalServer.mockResolvedValue(
      readinessServer({
        display_state: "ready",
        credential_state: "configured",
        tool_count: 1,
        reason_codes: [],
        primary_reason_code: null,
        allowed_actions: ["open_tool_catalog", "view_details"],
        message: "Server is ready."
      })
    )
    mocks.refreshExternalServerDiscovery.mockResolvedValue(
      readinessServer({
        display_state: "ready",
        credential_state: "configured",
        tool_count: 1,
        reason_codes: [],
        primary_reason_code: null,
        allowed_actions: ["open_tool_catalog", "view_details"],
        message: "Server is ready."
      })
    )
    mocks.updateExternalServerAuthTemplate.mockImplementation(async (_serverId: string, payload: unknown) => payload)
    mocks.setExternalServerSecret.mockResolvedValue({
      server_id: "docs-managed",
      secret_configured: true
    })
    mocks.setExternalServerSlotSecret.mockResolvedValue({
      server_id: "docs-managed",
      slot_name: "token_readonly",
      secret_configured: true
    })
    mocks.clearExternalServerSlotSecret.mockResolvedValue({ ok: true })
    mocks.importExternalServer.mockResolvedValue({
      id: "search-legacy",
      name: "Search Legacy",
      enabled: true,
      owner_scope_type: "global",
      transport: "websocket",
      config: {},
      secret_configured: false,
      server_source: "managed",
      binding_count: 0,
      runtime_executable: true,
      auth_template_present: false,
      auth_template_valid: false,
      auth_template_blocked_reason: "no_auth_template",
      credential_slots: []
    })
    mocks.createExternalServer.mockResolvedValue({
      id: "new-managed",
      name: "New Managed",
      enabled: true,
      owner_scope_type: "global",
      transport: "websocket",
      config: {},
      secret_configured: false,
      server_source: "managed",
      binding_count: 0,
      runtime_executable: true,
      auth_template_present: false,
      auth_template_valid: false,
      auth_template_blocked_reason: "no_auth_template",
      credential_slots: []
    })
    mocks.updateExternalServer.mockResolvedValue({
      id: "docs-managed",
      name: "Docs Managed Updated",
      enabled: true,
      owner_scope_type: "global",
      transport: "stdio",
      config: {},
      secret_configured: false,
      server_source: "managed",
      binding_count: 2,
      runtime_executable: true,
      auth_template_present: true,
      auth_template_valid: true,
      auth_template_blocked_reason: null,
      credential_slots: [
        {
          server_id: "docs-managed",
          slot_name: "token_readonly",
          display_name: "Read-only token",
          secret_kind: "bearer_token",
          privilege_class: "read",
          is_required: true,
          secret_configured: true
        }
      ]
    })
    mocks.refreshExternalServerDiscovery.mockResolvedValue({
      ok: true,
      status: "refreshed"
    })
    mocks.createExternalServerCredentialSlot.mockResolvedValue({
      server_id: "docs-managed",
      slot_name: "token_write",
      display_name: "Write token",
      secret_kind: "bearer_token",
      privilege_class: "write",
      is_required: false,
      secret_configured: false
    })
    mocks.updateExternalServerCredentialSlot.mockResolvedValue({
      server_id: "docs-managed",
      slot_name: "token_readonly",
      display_name: "Read-only token updated",
      secret_kind: "bearer_token",
      privilege_class: "read",
      is_required: true,
      secret_configured: true
    })
    mocks.deleteExternalServerCredentialSlot.mockResolvedValue({ ok: true })
    mocks.deleteExternalServer.mockResolvedValue({ ok: true })
    vi.stubGlobal("confirm", vi.fn(() => true))
  })

  afterEach(() => {
    Modal.destroyAll()
  })

  it("renders managed and legacy servers, supports import, and still saves managed secrets", async () => {
    const user = userEvent.setup()
    render(<ExternalServersTab />)

    expect((await screen.findAllByText(/legacy read only/i)).length).toBe(2)
    expect(screen.getByText("Search Legacy")).toBeTruthy()
    expect(screen.getByText("Docs Legacy")).toBeTruthy()
    expect(screen.getByText(/superseded by docs-managed/i)).toBeTruthy()
    expect(screen.getByText(/2 bindings/i)).toBeTruthy()
    expect(screen.getByText(/1 slot/i)).toBeTruthy()
    expect(screen.getAllByText("Read-only token").length).toBeGreaterThan(0)

    const secretInput = (await screen.findByLabelText(/slot secret/i)) as HTMLInputElement
    await user.type(secretInput, "super-secret")
    await user.click(screen.getByRole("button", { name: /save slot secret/i }))

    expect(mocks.setExternalServerSlotSecret).toHaveBeenCalledWith(
      "docs-managed",
      "token_readonly",
      "super-secret"
    )
    expect(await screen.findByText(/slot secret configured/i)).toBeTruthy()
    expect(secretInput.value).toBe("")
    expect(screen.queryByDisplayValue("super-secret")).toBeNull()

    await user.click(screen.getByRole("button", { name: /import to mcp hub/i }))
    expect(mocks.importExternalServer).toHaveBeenCalledWith("search-legacy")
  })

  it("renders no-auth local stdio servers as credential-complete", async () => {
    mocks.listExternalServers.mockResolvedValueOnce([
      {
        id: "toy-stdio",
        name: "Toy Stdio",
        enabled: true,
        owner_scope_type: "global",
        transport: "stdio",
        config: { command: "node", args: ["toy-server.mjs"], auth: { mode: "none" } },
        secret_configured: false,
        server_source: "managed",
        binding_count: 0,
        runtime_executable: true,
        auth_template_present: false,
        auth_template_valid: false,
        auth_template_blocked_reason: "no_auth_template",
        credential_slots: []
      }
    ])

    render(<ExternalServersTab />)

    expect((await screen.findAllByText("Toy Stdio")).length).toBeGreaterThan(0)
    expect((await screen.findAllByText(/no credentials required/i)).length).toBeGreaterThan(0)
    expect(screen.queryByText(/^No auth template$/i)).toBeNull()
    expect(screen.queryByText(/^no secret$/i)).toBeNull()
    expect(screen.queryByText(/legacy secret fallback/i)).toBeNull()
    expect(screen.queryByText(/add at least one credential slot/i)).toBeNull()
  })

  it("keeps the legacy server-level secret fallback when auth is still configured without slots", async () => {
    mocks.listExternalServers.mockResolvedValueOnce([
      {
        id: "docs-server-secret",
        name: "Docs Server Secret",
        enabled: true,
        owner_scope_type: "global",
        transport: "stdio",
        config: { command: "node", args: ["docs-server.mjs"], auth: { mode: "bearer_token", env: "DOCS_TOKEN" } },
        secret_configured: false,
        server_source: "managed",
        binding_count: 0,
        runtime_executable: true,
        auth_template_present: false,
        auth_template_valid: false,
        auth_template_blocked_reason: "no_auth_template",
        credential_slots: []
      }
    ])

    render(<ExternalServersTab />)

    expect((await screen.findAllByText("Docs Server Secret")).length).toBeGreaterThan(0)
    expect(await screen.findByText(/legacy secret fallback/i)).toBeTruthy()
    expect(screen.getByLabelText(/^secret$/i)).toBeTruthy()
    expect(screen.queryByText(/no credentials required/i)).toBeNull()
  })

  it("renders auth template readiness and saves transport-aware mappings", async () => {
    const user = userEvent.setup()
    render(<ExternalServersTab />)

    expect((await screen.findAllByText(/no auth template/i)).length).toBeGreaterThan(0)

    await user.click(screen.getByRole("button", { name: /add mapping/i }))
    await user.type(screen.getByLabelText(/target name 1/i), "DOCS_TOKEN")
    await user.type(screen.getByLabelText(/prefix 1/i), "Bearer ")
    await user.click(screen.getByRole("button", { name: /save auth template/i }))

    expect(mocks.updateExternalServerAuthTemplate).toHaveBeenCalledWith("docs-managed", {
      mode: "template",
      mappings: [
        {
          slot_name: "token_readonly",
          target_type: "env",
          target_name: "DOCS_TOKEN",
          prefix: "Bearer ",
          suffix: "",
          required: true
        }
      ]
    })
    expect(await screen.findByText(/auth template updated/i)).toBeTruthy()
  })

  it("preserves custom header targets for websocket auth templates", async () => {
    const user = userEvent.setup()
    mocks.listExternalServers.mockResolvedValueOnce([
      {
        id: "docs-websocket",
        name: "Docs Websocket",
        enabled: true,
        owner_scope_type: "global",
        transport: "websocket",
        config: {},
        secret_configured: false,
        server_source: "managed",
        binding_count: 1,
        runtime_executable: true,
        auth_template_present: false,
        auth_template_valid: false,
        auth_template_blocked_reason: "no_auth_template",
        credential_slots: [
          {
            server_id: "docs-websocket",
            slot_name: "token_readonly",
            display_name: "Read-only token",
            secret_kind: "bearer_token",
            privilege_class: "read",
            is_required: true,
            secret_configured: false
          }
        ]
      }
    ])
    mocks.getExternalServerAuthTemplate.mockResolvedValueOnce({
      mode: "template",
      mappings: []
    })

    render(<ExternalServersTab />)

    expect((await screen.findAllByText("Docs Websocket")).length).toBeGreaterThan(0)
    await user.click(screen.getByRole("button", { name: /add mapping/i }))
    await user.type(screen.getByLabelText(/target name 1/i), "X-DOCS-TOKEN")
    await user.type(screen.getByLabelText(/prefix 1/i), "Token ")
    await user.click(screen.getByRole("button", { name: /save auth template/i }))

    expect(mocks.updateExternalServerAuthTemplate).toHaveBeenCalledWith("docs-websocket", {
      mode: "template",
      mappings: [
        {
          slot_name: "token_readonly",
          target_type: "header",
          target_name: "X-DOCS-TOKEN",
          prefix: "Token ",
          suffix: "",
          required: true
        }
      ]
    })
  })

  it("creates, edits, and deletes managed servers and credential slots", async () => {
    const user = userEvent.setup()
    render(<ExternalServersTab />)

    await screen.findByText("Docs Legacy")

    await user.click(screen.getByRole("button", { name: /add slot/i }))
    await user.type(screen.getByLabelText(/slot name/i), "token_write")
    await user.type(screen.getByLabelText(/slot display name/i), "Write token")
    await user.selectOptions(screen.getByLabelText(/privilege class/i), "write")
    await user.click(screen.getByRole("button", { name: /^save slot$/i }))

    expect(mocks.createExternalServerCredentialSlot).toHaveBeenCalledWith("docs-managed", {
      slot_name: "token_write",
      display_name: "Write token",
      secret_kind: "bearer_token",
      privilege_class: "write",
      is_required: true
    })

    await user.click(screen.getByRole("button", { name: /edit read-only token/i }))
    const slotNameInput = screen.queryByLabelText(/slot name/i)
    expect(slotNameInput).toBeNull()
    const displayNameInput = screen.getByLabelText(/slot display name/i)
    await user.clear(displayNameInput)
    await user.type(displayNameInput, "Read-only token updated")
    await user.click(screen.getByRole("button", { name: /update slot/i }))

    expect(mocks.updateExternalServerCredentialSlot).toHaveBeenCalledWith("docs-managed", "token_readonly", {
      display_name: "Read-only token updated",
      secret_kind: "bearer_token",
      privilege_class: "read",
      is_required: true
    })

    await user.click(screen.getByRole("button", { name: /delete read-only token/i }))
    await user.click(screen.getByRole("button", { name: /^Delete$/i }))
    expect(mocks.deleteExternalServerCredentialSlot).toHaveBeenCalledWith("docs-managed", "token_readonly")

    await user.click(screen.getByRole("button", { name: /new managed server/i }))
    await user.type(screen.getByLabelText(/server id/i), "new-managed")
    await user.type(screen.getByLabelText(/^name$/i), "New Managed")
    await user.selectOptions(screen.getByRole("combobox", { name: /^transport$/i }), "websocket")
    await user.click(screen.getByRole("button", { name: /save server/i }))

    expect(mocks.createExternalServer).toHaveBeenCalledWith({
      server_id: "new-managed",
      name: "New Managed",
      transport: "websocket",
      config: {},
      owner_scope_type: "global",
      enabled: true
    })

    await user.click(screen.getByRole("button", { name: /edit docs managed/i }))
    const nameInput = screen.getByLabelText(/^name$/i)
    await user.clear(nameInput)
    await user.type(nameInput, "Docs Managed Updated")
    await user.click(screen.getByRole("button", { name: /update server/i }))

    expect(mocks.updateExternalServer).toHaveBeenCalledWith("docs-managed", {
      name: "Docs Managed Updated",
      transport: "stdio",
      config: {},
      owner_scope_type: "global",
      enabled: true
    })

    await user.click(screen.getByRole("button", { name: /delete docs managed/i }))
    await user.click(screen.getByRole("button", { name: /^Delete$/i }))
    expect(mocks.deleteExternalServer).toHaveBeenCalledWith("docs-managed")
  }, 15000)

  it("refreshes external discovery and MCP tool queries after managed server mutations", async () => {
    const user = userEvent.setup()
    render(<ExternalServersTab />)

    await screen.findByText("Docs Legacy")

    await user.click(screen.getByRole("button", { name: /new managed server/i }))
    await user.type(screen.getByLabelText(/server id/i), "new-managed")
    await user.type(screen.getByLabelText(/^name$/i), "New Managed")
    await user.selectOptions(screen.getByRole("combobox", { name: /^transport$/i }), "websocket")
    await user.click(screen.getByRole("button", { name: /save server/i }))

    await waitFor(() => {
      expect(mocks.createExternalServer).toHaveBeenCalled()
    })
    expect(mocks.refreshExternalServerDiscovery).toHaveBeenLastCalledWith("new-managed")
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-tools"] })
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-tool-catalogs"] })
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-tool-modules"] })
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-health"] })
    expect(await screen.findByText(/server created and tools refreshed/i)).toBeTruthy()

    await user.click(screen.getByRole("button", { name: /edit docs managed/i }))
    const nameInput = screen.getByLabelText(/^name$/i)
    await user.clear(nameInput)
    await user.type(nameInput, "Docs Managed Updated")
    await user.click(screen.getByRole("button", { name: /update server/i }))

    await waitFor(() => {
      expect(mocks.updateExternalServer).toHaveBeenCalledWith(
        "docs-managed",
        expect.objectContaining({ enabled: true })
      )
    })
    expect(mocks.refreshExternalServerDiscovery).toHaveBeenLastCalledWith("docs-managed")
    expect(await screen.findByText(/server updated and tools refreshed/i)).toBeTruthy()

    await user.click(screen.getByRole("button", { name: /edit docs managed/i }))
    await user.click(screen.getByRole("checkbox", { name: /enabled/i }))
    await user.click(screen.getByRole("button", { name: /update server/i }))

    await waitFor(() => {
      expect(mocks.updateExternalServer).toHaveBeenLastCalledWith(
        "docs-managed",
        expect.objectContaining({ enabled: false })
      )
    })
    expect(mocks.refreshExternalServerDiscovery).toHaveBeenLastCalledWith()

    await user.click(screen.getByRole("button", { name: /import to mcp hub/i }))
    await waitFor(() => {
      expect(mocks.importExternalServer).toHaveBeenCalledWith("search-legacy")
    })
    expect(mocks.refreshExternalServerDiscovery).toHaveBeenLastCalledWith("search-legacy")
    expect(await screen.findByText(/legacy server imported and tools refreshed/i)).toBeTruthy()

    await user.click(screen.getByRole("button", { name: /delete docs managed/i }))
    await user.click(await screen.findByRole("button", { name: /^delete$/i }))

    await waitFor(() => {
      expect(mocks.deleteExternalServer).toHaveBeenCalledWith("docs-managed")
    })
    expect(mocks.refreshExternalServerDiscovery).toHaveBeenLastCalledWith()
    expect(await screen.findByText(/server deleted and tools refreshed/i)).toBeTruthy()
  }, 20000)

  it("keeps persistence success visible when runtime discovery refresh fails", async () => {
    const user = userEvent.setup()
    mocks.refreshExternalServerDiscovery.mockRejectedValueOnce(new Error("runtime unavailable"))
    render(<ExternalServersTab />)

    expect((await screen.findAllByText("Docs Managed")).length).toBeGreaterThan(0)
    await user.click(screen.getByRole("button", { name: /new managed server/i }))
    await user.type(screen.getByLabelText(/server id/i), "new-managed")
    await user.type(screen.getByLabelText(/^name$/i), "New Managed")
    await user.click(screen.getByRole("button", { name: /save server/i }))

    expect(await screen.findByText(/server created, but discovery refresh failed/i)).toBeTruthy()
    expect(screen.getByText(/retry runtime refresh/i)).toBeTruthy()
    expect(mocks.listExternalServers).toHaveBeenCalledTimes(2)
  })

  it("keeps persistence success visible when runtime discovery resolves with errors", async () => {
    const user = userEvent.setup()
    mocks.refreshExternalServerDiscovery.mockResolvedValueOnce({
      ok: false,
      message: "Discovery refreshed with errors",
      errors: { "new-managed": "external_server_discovery_failed" }
    })
    render(<ExternalServersTab />)

    expect((await screen.findAllByText("Docs Managed")).length).toBeGreaterThan(0)
    await user.click(screen.getByRole("button", { name: /new managed server/i }))
    await user.type(screen.getByLabelText(/server id/i), "new-managed")
    await user.type(screen.getByLabelText(/^name$/i), "New Managed")
    await user.click(screen.getByRole("button", { name: /save server/i }))

    expect(await screen.findByText(/server created, but discovery refresh failed/i)).toBeTruthy()
    expect(screen.getByText(/external_server_discovery_failed/i)).toBeTruthy()
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-tools"] })
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-tool-catalogs"] })
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-tool-modules"] })
    expect(mocks.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcp-health"] })
    expect(mocks.listExternalServers).toHaveBeenCalledTimes(2)
  })

  it("opens the managed server editor from a drill target", async () => {
    const onDrillHandled = vi.fn()
    render(
      <ExternalServersTab
        drillTarget={{
          tab: "credentials",
          object_kind: "external_server",
          object_id: "docs-managed",
          action: "edit",
          request_id: 12
        }}
        onDrillHandled={onDrillHandled}
      />
    )

    expect(await screen.findByDisplayValue("Docs Managed")).toBeTruthy()
    expect(screen.getByDisplayValue("stdio")).toBeTruthy()
    expect(onDrillHandled).toHaveBeenCalledWith(12)
  })

  it("falls back to visible focus for legacy servers from a drill target", async () => {
    const onDrillHandled = vi.fn()
    render(
      <ExternalServersTab
        drillTarget={{
          tab: "credentials",
          object_kind: "external_server",
          object_id: "search-legacy",
          action: "focus",
          request_id: 13
        }}
        onDrillHandled={onDrillHandled}
      />
    )

    expect(await screen.findByText("Search Legacy")).toBeTruthy()
    expect(await screen.findByText(/focused from audit/i)).toBeTruthy()
    expect(onDrillHandled).toHaveBeenCalledWith(13)
  })

  it("renders no-auth managed stdio rows as not requiring credentials", async () => {
    mocks.listExternalServers.mockResolvedValueOnce([
      {
        id: "local-docs",
        name: "Local Docs",
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
        credential_slots: []
      }
    ])
    mocks.getMcpHubReadiness.mockResolvedValueOnce(
      readinessResponse([
        readinessServer({
          server_id: "local-docs",
          server_name: "Local Docs",
          credential_state: "not_required",
          reason_codes: ["discovery_not_run"],
          primary_reason_code: "discovery_not_run",
          allowed_actions: ["refresh_discovery", "edit_config"],
          message: "Run discovery to populate this server's tool catalog. No credentials required."
        })
      ])
    )

    render(<ExternalServersTab />)

    const row = await findServerRow("Local Docs")
    expect(within(row).getAllByText(/no credentials required/i).length).toBeGreaterThan(0)
    expect(within(row).queryByText(/^no secret$/i)).toBeNull()
    expect(within(row).queryByText(/^no auth template$/i)).toBeNull()
  })

  it("renders legacy secret fallback only for managed rows that use server-level secrets", async () => {
    mocks.listExternalServers.mockResolvedValueOnce([
      {
        id: "local-docs",
        name: "Local Docs",
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
        credential_slots: []
      },
      {
        id: "legacy-secret-managed",
        name: "Legacy Secret Managed",
        enabled: true,
        owner_scope_type: "global",
        transport: "websocket",
        config: {},
        secret_configured: true,
        server_source: "managed",
        binding_count: 0,
        runtime_executable: true,
        auth_template_present: false,
        auth_template_valid: false,
        auth_template_blocked_reason: "no_auth_template",
        credential_slots: []
      }
    ])
    mocks.getMcpHubReadiness.mockResolvedValueOnce(
      readinessResponse([
        readinessServer({
          server_id: "local-docs",
          server_name: "Local Docs",
          credential_state: "not_required",
          reason_codes: ["discovery_not_run"],
          primary_reason_code: "discovery_not_run"
        }),
        readinessServer({
          server_id: "legacy-secret-managed",
          server_name: "Legacy Secret Managed",
          credential_state: "legacy_fallback",
          reason_codes: ["discovery_not_run"],
          primary_reason_code: "discovery_not_run",
          message: "Run discovery to populate this server's tool catalog. Legacy Secret Fallback is configured."
        })
      ])
    )

    render(<ExternalServersTab />)

    const noAuthRow = await findServerRow("Local Docs")
    const fallbackRow = await findServerRow("Legacy Secret Managed")
    expect(within(noAuthRow).queryByText(/legacy secret fallback/i)).toBeNull()
    expect(within(fallbackRow).getAllByText(/legacy secret fallback/i).length).toBeGreaterThan(0)
  })

  it("uses readiness and registry data so ready rows do not look undiscovered", async () => {
    mocks.listExternalServers.mockResolvedValueOnce([
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
        auth_template_present: true,
        auth_template_valid: true,
        auth_template_blocked_reason: null,
        credential_slots: []
      }
    ])
    mocks.getToolRegistrySummary.mockResolvedValueOnce({
      entries: [toolEntry("docs-managed")],
      modules: []
    })
    mocks.getMcpHubReadiness.mockResolvedValueOnce(
      readinessResponse([
        readinessServer({
          display_state: "ready",
          credential_state: "configured",
          tool_count: 1,
          reason_codes: [],
          primary_reason_code: null,
          allowed_actions: ["open_tool_catalog", "view_details"],
          message: "Server is ready with 1 available tool. Credentials are configured."
        })
      ])
    )

    render(<ExternalServersTab />)

    const row = await findServerRow("Docs Managed")
    expect(within(row).getByText(/^Ready$/i)).toBeTruthy()
    expect(within(row).getByText(/1 tool/i)).toBeTruthy()
    expect(within(row).queryByText(/discovery_not_run/i)).toBeNull()
    expect(within(row).queryByText(/run discovery/i)).toBeNull()
  })

  it("shows stale refresh copy when backend readiness reports config changes", async () => {
    mocks.getMcpHubReadiness.mockResolvedValueOnce(
      readinessResponse([
        readinessServer({
          display_state: "stale",
          credential_state: "configured",
          tool_count: 2,
          reason_codes: ["config_changed"],
          primary_reason_code: "config_changed",
          allowed_actions: ["refresh_discovery", "edit_config", "view_details"],
          message: "Configuration changed after the last discovery run."
        })
      ])
    )

    render(<ExternalServersTab />)

    const row = await findServerRow("Docs Managed")
    expect(within(row).getByText(/^Stale$/i)).toBeTruthy()
    expect(within(row).getByText(/configuration changed/i)).toBeTruthy()
    expect(within(row).getByRole("button", { name: /refresh tools/i })).toBeTruthy()
  })

  it("wires readiness recovery actions to real handlers", async () => {
    const user = userEvent.setup()
    const onOpenToolCatalog = vi.fn()
    mocks.getMcpHubReadiness.mockResolvedValue(
      readinessResponse([
        readinessServer({
          credential_state: "required_missing",
          reason_codes: ["auth_missing", "config_changed"],
          primary_reason_code: "auth_missing",
          allowed_actions: [
            "validate",
            "refresh_discovery",
            "edit_config",
            "open_credentials",
            "view_details",
            "open_tool_catalog"
          ],
          message: "Credentials are required before this server can be used."
        })
      ])
    )

    render(<ExternalServersTab onOpenToolCatalog={onOpenToolCatalog} />)

    const row = await findServerRow("Docs Managed")

    await user.click(within(row).getByRole("button", { name: /validate/i }))
    expect(mocks.validateExternalServer).toHaveBeenCalledWith("docs-managed")

    await user.click(within(row).getByRole("button", { name: /refresh tools/i }))
    expect(mocks.refreshExternalServerDiscovery).toHaveBeenCalledWith("docs-managed")
    expect(mocks.listExternalServers).toHaveBeenCalledTimes(3)

    await user.click(within(row).getByRole("button", { name: /details/i }))
    expect(await screen.findByText(/server id: docs-managed/i)).toBeTruthy()
    const closeButtons = screen.getAllByRole("button", { name: /^close$/i })
    await user.click(closeButtons[closeButtons.length - 1])

    await user.click(within(row).getByRole("button", { name: /edit config/i }))
    expect(await screen.findByLabelText(/^name$/i)).toHaveValue("Docs Managed")

    await user.click(within(row).getByRole("button", { name: /^credentials$/i }))
    expect(screen.getByLabelText(/slot secret/i)).toBeTruthy()

    await user.click(within(row).getByRole("button", { name: /tool catalog/i }))
    expect(onOpenToolCatalog).toHaveBeenCalledTimes(1)
  }, 15000)
})
