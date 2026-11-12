// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

const mocks = vi.hoisted(() => ({
  listExternalServers: vi.fn(),
  setExternalServerSecret: vi.fn(),
  setExternalServerSlotSecret: vi.fn(),
  clearExternalServerSlotSecret: vi.fn(),
  getExternalServerAuthTemplate: vi.fn(),
  getMcpHubReadiness: vi.fn(),
  getToolRegistrySummary: vi.fn(),
  refreshExternalServerDiscovery: vi.fn(),
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

vi.mock("@/services/tldw/mcp-hub", () => ({
  listExternalServers: (...args: unknown[]) => mocks.listExternalServers(...args),
  setExternalServerSecret: (...args: unknown[]) => mocks.setExternalServerSecret(...args),
  setExternalServerSlotSecret: (...args: unknown[]) => mocks.setExternalServerSlotSecret(...args),
  clearExternalServerSlotSecret: (...args: unknown[]) => mocks.clearExternalServerSlotSecret(...args),
  getExternalServerAuthTemplate: (...args: unknown[]) => mocks.getExternalServerAuthTemplate(...args),
  getMcpHubReadiness: (...args: unknown[]) => mocks.getMcpHubReadiness(...args),
  getToolRegistrySummary: (...args: unknown[]) => mocks.getToolRegistrySummary(...args),
  refreshExternalServerDiscovery: (...args: unknown[]) => mocks.refreshExternalServerDiscovery(...args),
  refreshExternalServerReadinessDiscovery: (...args: unknown[]) => mocks.refreshExternalServerDiscovery(...args),
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

const withClearedDiagnosticEnv = async (fn: () => Promise<void>) => {
  const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  const originalApiUrl = process.env.NEXT_PUBLIC_API_URL

  try {
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    delete process.env.NEXT_PUBLIC_API_URL
    await fn()
  } finally {
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }

    if (originalApiUrl === undefined) {
      delete process.env.NEXT_PUBLIC_API_URL
    } else {
      process.env.NEXT_PUBLIC_API_URL = originalApiUrl
    }
  }
}

describe("ExternalServersTab", () => {
  beforeEach(() => {
    vi.clearAllMocks()
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

  it("uses header targets for SSE auth templates", async () => {
    const user = userEvent.setup()
    mocks.listExternalServers.mockResolvedValueOnce([
      {
        id: "docs-sse",
        name: "Docs SSE",
        enabled: true,
        owner_scope_type: "global",
        transport: "sse",
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
            server_id: "docs-sse",
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

    expect((await screen.findAllByText("Docs SSE")).length).toBeGreaterThan(0)
    expect(screen.getByText(/template target: header/i)).toBeTruthy()
    expect(screen.queryByText(/unsupported transport/i)).toBeNull()

    await user.click(screen.getByRole("button", { name: /add mapping/i }))
    await user.type(screen.getByLabelText(/target name 1/i), "Authorization")
    await user.type(screen.getByLabelText(/prefix 1/i), "Bearer ")
    await user.click(screen.getByRole("button", { name: /save auth template/i }))

    expect(mocks.updateExternalServerAuthTemplate).toHaveBeenCalledWith("docs-sse", {
      mode: "template",
      mappings: [
        {
          slot_name: "token_readonly",
          target_type: "header",
          target_name: "Authorization",
          prefix: "Bearer ",
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
    await user.click(screen.getByRole("button", { name: /advanced\/manual/i }))
    await user.type(screen.getByLabelText(/server id/i), "new-managed")
    await user.type(screen.getByLabelText(/^name$/i), "New Managed")
    await user.selectOptions(screen.getByLabelText(/transport/i), "websocket")
    await user.click(screen.getByRole("button", { name: /save without discovery/i }))

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

  it("shows guided setup choices before managed server fields", async () => {
    const user = userEvent.setup()
    render(<ExternalServersTab />)

    await screen.findByText("Docs Legacy")
    await user.click(screen.getByRole("button", { name: /new managed server/i }))

    expect(screen.getByRole("button", { name: /local stdio/i })).toBeTruthy()
    expect(screen.getByRole("button", { name: /http\/sse/i })).toBeTruthy()
    expect(screen.getByRole("button", { name: /import config/i })).toBeTruthy()
    expect(screen.getByRole("button", { name: /advanced\/manual/i })).toBeTruthy()
    expect(screen.queryByLabelText(/config json/i)).toBeNull()
  })

  it("keeps the raw config create path behind Advanced/manual", async () => {
    const user = userEvent.setup()
    render(<ExternalServersTab />)

    await screen.findByText("Docs Legacy")
    await user.click(screen.getByRole("button", { name: /new managed server/i }))
    await user.click(screen.getByRole("button", { name: /advanced\/manual/i }))

    await user.type(screen.getByLabelText(/server id/i), "manual-managed")
    await user.type(screen.getByLabelText(/^name$/i), "Manual Managed")
    await user.selectOptions(screen.getByLabelText(/transport/i), "websocket")
    await user.click(screen.getByRole("button", { name: /save without discovery/i }))

    expect(mocks.createExternalServer).toHaveBeenCalledWith({
      server_id: "manual-managed",
      name: "Manual Managed",
      transport: "websocket",
      config: {},
      owner_scope_type: "global",
      enabled: true
    })
    expect(mocks.refreshExternalServerDiscovery).not.toHaveBeenCalled()
  })

  it("previews pasted managed config and blocks invalid import JSON", async () => {
    const user = userEvent.setup()
    render(<ExternalServersTab />)

    await screen.findByText("Docs Legacy")
    await user.click(screen.getByRole("button", { name: /new managed server/i }))
    await user.click(screen.getByRole("button", { name: /import config/i }))

    await user.type(screen.getByLabelText(/managed config json/i), "not json")
    await user.click(screen.getByRole("button", { name: /save and discover tools/i }))

    expect(await screen.findByText(/import config json must be valid json/i)).toBeTruthy()
    expect(mocks.createExternalServer).not.toHaveBeenCalled()

    await user.clear(screen.getByLabelText(/managed config json/i))
    fireEvent.change(screen.getByLabelText(/managed config json/i), {
      target: {
        value: JSON.stringify({
          server_id: "docs-http",
          name: "Docs HTTP",
          transport: "sse",
          config: {
            url: "https://example.test/mcp"
          }
        })
      }
    })

    expect(await screen.findByText(/preview: docs http/i)).toBeTruthy()
    expect(screen.getByText(/transport: sse/i)).toBeTruthy()
  })

  it("saves a guided stdio server, discovers tools, and shows next actions", async () => {
    const user = userEvent.setup()
    const onOpenToolCatalog = vi.fn()
    mocks.createExternalServer.mockResolvedValueOnce({
      id: "docs-local",
      name: "Docs Local",
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
    })
    render(<ExternalServersTab onOpenToolCatalog={onOpenToolCatalog} />)

    await screen.findByText("Docs Legacy")
    await user.click(screen.getByRole("button", { name: /new managed server/i }))
    await user.click(screen.getByRole("button", { name: /local stdio/i }))
    await user.type(screen.getByLabelText(/server id/i), "docs-local")
    await user.type(screen.getByLabelText(/^name$/i), "Docs Local")
    await user.type(screen.getByLabelText(/command/i), "uvx")
    await user.type(screen.getByLabelText(/args/i), "mcp-server-docs --stdio")
    await user.type(screen.getByLabelText(/working directory/i), "/tmp/docs")
    await user.type(screen.getByLabelText(/env vars/i), "DOCS_MODE=readonly")
    await user.click(screen.getByRole("button", { name: /save and discover tools/i }))

    expect(mocks.createExternalServer).toHaveBeenCalledWith({
      server_id: "docs-local",
      name: "Docs Local",
      transport: "stdio",
      config: {
        stdio: {
          command: "uvx",
          args: ["mcp-server-docs", "--stdio"],
          cwd: "/tmp/docs",
          env: {
            DOCS_MODE: "readonly"
          }
        }
      },
      owner_scope_type: "global",
      enabled: true
    })
    expect(mocks.refreshExternalServerDiscovery).toHaveBeenCalledWith("docs-local")
    expect(await screen.findByText(/docs local saved/i)).toBeTruthy()

    await user.click(screen.getByRole("button", { name: /tool catalog/i }))
    expect(onOpenToolCatalog).toHaveBeenCalledTimes(1)
  }, 15000)

  it("keeps unsupported HTTP/SSE unavailable without creating or discovering a server", async () => {
    const user = userEvent.setup()
    render(<ExternalServersTab />)

    await screen.findByText("Docs Legacy")
    await user.click(screen.getByRole("button", { name: /new managed server/i }))
    const choice = screen.getByRole("button", { name: /http\/sse/i })
    expect(choice).toBeDisabled()
    await user.click(choice)

    expect(screen.getByText(/HTTP\/SSE is unavailable.*stdio.*WebSocket/i)).toBeVisible()
    expect(screen.queryByLabelText(/^url$/i)).toBeNull()
    expect(mocks.createExternalServer).not.toHaveBeenCalled()
    expect(mocks.refreshExternalServerDiscovery).not.toHaveBeenCalled()
  }, 15000)

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

    expect(await screen.findByRole("textbox", { name: "Name" })).toHaveValue("Docs Managed")
    expect(screen.getByRole("combobox", { name: "Transport" })).toHaveValue("stdio")
    expect(onDrillHandled).toHaveBeenCalledWith(12)
  })

  it("focuses a managed server without opening config from a focus drill target", async () => {
    const onDrillHandled = vi.fn()
    render(
      <ExternalServersTab
        drillTarget={{
          tab: "credentials",
          object_kind: "external_server",
          object_id: "docs-managed",
          action: "focus",
          request_id: 14
        }}
        onDrillHandled={onDrillHandled}
      />
    )

    expect(await screen.findByText(/focused from audit/i)).toBeTruthy()
    expect(screen.queryByLabelText(/^name$/i)).toBeNull()
    expect(onDrillHandled).toHaveBeenCalledWith(14)
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
    expect(screen.getAllByText(/no credentials required/i).length).toBeGreaterThan(1)
    expect(screen.queryByText(/^No auth template$/i)).toBeNull()
    expect(screen.queryByText(/^Legacy Secret Fallback$/i)).toBeNull()
    expect(screen.queryByLabelText(/^Secret$/i)).toBeNull()
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

    await userEvent.setup().selectOptions(screen.getByLabelText(/^Server$/i), "legacy-secret-managed")

    expect(screen.getAllByText(/^Legacy Secret Fallback$/i).length).toBeGreaterThan(1)
    expect(screen.getByLabelText(/^Secret$/i)).toBeTruthy()
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

  it("keeps server inventory visible when readiness metadata fails to load", async () => {
    mocks.getMcpHubReadiness.mockRejectedValueOnce(new Error("readiness unavailable"))

    render(<ExternalServersTab />)

    const row = await findServerRow("Docs Managed")
    expect(within(row).getByText(/credentials required/i)).toBeTruthy()
    expect(await screen.findByText(/readiness details are limited/i)).toBeTruthy()
    expect(screen.queryByText(/no external servers configured/i)).toBeNull()
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

  it("opens diagnostics with readiness, environment, and redacted setup context", async () => {
    await withClearedDiagnosticEnv(async () => {
      const user = userEvent.setup()
      mocks.listExternalServers.mockResolvedValueOnce([
        {
          id: "docs-managed",
          name: "Docs Managed",
          enabled: true,
          owner_scope_type: "global",
          transport: "stdio",
          config: {
            command: "uvx",
            args: ["mcp-docs", "--token", "arg-secret", "--api-key=arg-key", "--mode", "readonly"],
            env: {
              API_TOKEN: "env-secret",
              MODE: "readonly"
            },
            headers: {
              Authorization: "Bearer header-secret",
              "X-Trace": "trace-id"
            },
            endpoint: "https://example.test/mcp?api_key=url-secret&mode=readonly",
            nested: [{ password: "nested-secret", label: "diagnostic-safe" }]
          },
          secret_configured: false,
          server_source: "managed",
          binding_count: 1,
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
            display_state: "needs_attention",
            credential_state: "required_missing",
            tool_count: 2,
            reason_codes: ["auth_missing", "discovery_failed"],
            primary_reason_code: "auth_missing",
            allowed_actions: ["open_credentials", "refresh_discovery", "view_details"],
            message: "Credentials are required before this server can be used.",
            current_operation: {
              operation_type: "discovery",
              started_at: "2026-06-27T08:00:00Z",
              message: "Refreshing catalog"
            },
            last_validation_at: "2026-06-27T07:00:00Z",
            last_discovery_at: "2026-06-27T07:10:00Z",
            last_successful_discovery_at: "2026-06-27T06:00:00Z",
            last_error_category: "auth",
            last_error_message: "Missing token"
          })
        ])
      )

      render(<ExternalServersTab />)

      const row = await findServerRow("Docs Managed")
      await user.click(within(row).getByRole("button", { name: /details/i }))

      const diagnosticsTitle = await screen.findByText(/docs managed readiness details/i)
      const diagnosticsDialog = diagnosticsTitle.closest(".ant-modal") as HTMLElement | null
      if (!diagnosticsDialog) {
        throw new Error("Expected diagnostics modal container")
      }
      const diagnostics = within(diagnosticsDialog)
      expect(await diagnostics.findByText(/display state: needs_attention/i)).toBeTruthy()
      expect(diagnostics.getByText(/primary reason: auth_missing/i)).toBeTruthy()
      expect(diagnostics.getByText(/reason codes: auth_missing, discovery_failed/i)).toBeTruthy()
      expect(diagnostics.getByText(/credential state: required_missing/i)).toBeTruthy()
      expect(diagnostics.getByText(/transport: stdio/i)).toBeTruthy()
      expect(diagnostics.getByText(/tools: 2/i)).toBeTruthy()
      expect(diagnostics.getByText(/last stored validation: 2026-06-27T07:00:00Z/i)).toBeTruthy()
      expect(diagnostics.getByText(/last discovery: 2026-06-27T07:10:00Z/i)).toBeTruthy()
      expect(diagnostics.getByText(/last successful discovery: 2026-06-27T06:00:00Z/i)).toBeTruthy()
      expect(diagnostics.getByText(/current operation: discovery since 2026-06-27T08:00:00Z, Refreshing catalog/i)).toBeTruthy()
      expect(diagnostics.getByText(/last error category: auth/i)).toBeTruthy()
      expect(diagnostics.getByText(/last error message: Missing token/i)).toBeTruthy()
      expect(diagnostics.getByText(/deployment mode: Not available in this client/i)).toBeTruthy()
      expect(diagnostics.getByText(/api origin: Not available in this client/i)).toBeTruthy()
      expect(diagnostics.getByText(/health endpoint: Not available in this client/i)).toBeTruthy()
      expect(diagnostics.getByText(/latest health result: Not available in this client/i)).toBeTruthy()
      expect(diagnostics.getByText(/audit details: use the governance audit tab/i)).toBeTruthy()
      expect(diagnostics.getByText(/use an isolated test database and temporary MCP server config/i)).toBeTruthy()

      const diagnosticsConfig = diagnostics.getByTestId("mcp-server-diagnostics-config").textContent ?? ""
      expect(diagnosticsConfig).toContain("[redacted]")
      expect(diagnosticsConfig).toContain('"MODE": "readonly"')
      expect(diagnosticsConfig).toContain('"X-Trace": "trace-id"')
      expect(diagnosticsConfig).toContain('"label": "diagnostic-safe"')
      expect(diagnosticsConfig).not.toContain("arg-secret")
      expect(diagnosticsConfig).not.toContain("arg-key")
      expect(diagnosticsConfig).not.toContain("env-secret")
      expect(diagnosticsConfig).not.toContain("header-secret")
      expect(diagnosticsConfig).not.toContain("url-secret")
      expect(diagnosticsConfig).not.toContain("nested-secret")
    })
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

    await user.click(within(row).getByRole("button", { name: /check readiness/i }))
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
  describe("credential context", () => {
    const configureServers = async (legacy = false) => {
      const [first] = await mocks.listExternalServers()
      const slots = legacy ? [] : [
        first.credential_slots[0],
        { ...first.credential_slots[0], slot_name: "token_write", display_name: "Write token" }
      ]
      const rows = [
        { ...first, credential_slots: slots },
        {
          ...first,
          id: "docs-second",
          name: "Docs Second",
          credential_slots: slots.map((slot) => ({ ...slot, server_id: "docs-second" }))
        }
      ]
      mocks.listExternalServers.mockClear()
      mocks.listExternalServers.mockResolvedValue(rows)
      mocks.getMcpHubReadiness.mockResolvedValue(readinessResponse([
        readinessServer({}),
        readinessServer({ server_id: "docs-second", server_name: "Docs Second" })
      ]))
      return rows
    }

    const findCredentialInput = async (legacy = false) => {
      const input = await screen.findByLabelText(legacy ? /^secret$/i : /^slot secret$/i)
      if (!legacy) await waitFor(() =>
        expect(screen.getByLabelText(/^credential slot$/i)).toHaveValue("token_readonly")
      )
      return input
    }

    it.each(["server", "slot"])("retires a slot draft when its %s changes", async (target) => {
      await configureServers()
      render(<ExternalServersTab />)
      const input = await findCredentialInput()
      fireEvent.change(input, { target: { value: "first-target-draft" } })
      fireEvent.change(screen.getByLabelText(target === "server" ? /^server$/i : /^credential slot$/i), {
        target: { value: target === "server" ? "docs-second" : "token_write" }
      })
      expect(input).toHaveValue("")
      expect(screen.getByRole("button", { name: /^save slot secret$/i })).toBeDisabled()
      expect(mocks.setExternalServerSlotSecret).not.toHaveBeenCalled()
    })

    it("retires a legacy draft when its server changes", async () => {
      await configureServers(true)
      render(<ExternalServersTab />)
      const input = await findCredentialInput(true)
      fireEvent.change(input, { target: { value: "first-target-draft" } })
      fireEvent.change(screen.getByLabelText(/^server$/i), { target: { value: "docs-second" } })
      expect(input).toHaveValue("")
      expect(screen.getByRole("button", { name: /^save secret$/i })).toBeDisabled()
      expect(mocks.setExternalServerSecret).not.toHaveBeenCalled()
    })

    it.each([
      ["legacy", "resolve"], ["legacy", "reject"],
      ["slot", "resolve"], ["slot", "reject"],
      ["clear", "resolve"], ["clear", "reject"]
    ])("retires an old %s %s after drilldown leaves and returns to the same server", async (kind, outcome) => {
      await configureServers(kind === "legacy")
      let resolve!: (value: unknown) => void
      let reject!: (reason: Error) => void
      const pending = new Promise((yes, no) => { resolve = yes; reject = no })
      const operation = kind === "legacy" ? mocks.setExternalServerSecret
        : kind === "slot" ? mocks.setExternalServerSlotSecret : mocks.clearExternalServerSlotSecret
      operation.mockReturnValueOnce(pending)
      const view = render(<ExternalServersTab />)
      const inputName = kind === "legacy" ? /^secret$/i : /^slot secret$/i
      const input = await findCredentialInput(kind === "legacy")
      if (kind !== "clear") fireEvent.change(input, { target: { value: "first-target-draft" } })
      fireEvent.click(screen.getByRole("button", { name: kind === "legacy" ? /^save secret$/i
        : kind === "slot" ? /^save slot secret$/i : /^clear slot secret$/i }))
      try {
        expect(operation).toHaveBeenCalledTimes(1)
        expect(operation).toHaveBeenCalledWith(...(kind === "legacy"
          ? ["docs-managed", "first-target-draft"]
          : kind === "slot" ? ["docs-managed", "token_readonly", "first-target-draft"]
            : ["docs-managed", "token_readonly"]))
        for (const [object_id, request_id] of [["docs-second", 41], ["docs-managed", 42]] as const) {
          view.rerender(<ExternalServersTab drillTarget={{
            tab: "credentials", object_kind: "external_server", object_id, request_id, action: "focus"
          }} />)
          expect(screen.getByLabelText(/^server$/i)).toHaveValue(object_id)
        }
      } finally {
        await act(async () => {
          if (outcome === "resolve") resolve({ ok: true })
          else reject(new Error("old-target-rejected"))
          await pending.catch(() => undefined)
        })
      }
      expect(screen.getByLabelText(inputName)).toHaveValue("")
      expect(screen.queryByText(/^(Secret configured|Slot secret configured|Slot secret cleared)$/)).toBeNull()
      expect(screen.queryByText(/old-target-rejected|Failed to (save|clear).*secret/i)).toBeNull()
      expect(mocks.listExternalServers).toHaveBeenCalledTimes(1)
    })

    it("locks the credential draft during its save and keeps the current selection after inventory reload", async () => {
      const rows = await configureServers()
      let finish!: (value: unknown) => void
      const pending = new Promise((resolve) => { finish = resolve })
      mocks.listExternalServers.mockResolvedValueOnce(rows).mockReturnValueOnce(pending)
      render(<ExternalServersTab />)
      const input = await findCredentialInput()
      fireEvent.change(input, { target: { value: "first-target-draft" } })
      fireEvent.click(screen.getByRole("button", { name: /^save slot secret$/i }))
      try {
        expect(input).toBeDisabled()
        expect(screen.getByRole("button", { name: /^clear slot secret$/i })).toBeDisabled()
        await screen.findByText(/^Slot secret configured$/)
        fireEvent.change(screen.getByLabelText(/^server$/i), { target: { value: "docs-second" } })
      } finally {
        await act(async () => { finish(rows); await pending })
      }
      expect(screen.getByLabelText(/^server$/i)).toHaveValue("docs-second")
      expect(screen.getByLabelText(/^slot secret$/i)).toHaveValue("")
      expect(screen.getByLabelText(/^slot secret$/i)).toBeEnabled()
      expect(screen.queryByText(/^Slot secret configured$/)).toBeNull()
    })

    it("keeps the latest selected server when an inventory refresh removes the previous one", async () => {
      const rows = await configureServers()
      const refreshed = [{ ...rows[0], id: "docs-third", name: "Docs Third" }, rows[1]]
      let finish!: (value: unknown) => void
      const pending = new Promise((resolve) => { finish = resolve })
      mocks.listExternalServers.mockResolvedValueOnce(rows).mockReturnValueOnce(pending)
      render(<ExternalServersTab />)
      const input = await findCredentialInput()
      fireEvent.change(input, { target: { value: "first-target-draft" } })
      fireEvent.click(screen.getByRole("button", { name: /^save slot secret$/i }))
      try {
        await screen.findByText(/^Slot secret configured$/)
        fireEvent.change(screen.getByLabelText(/^server$/i), { target: { value: "docs-second" } })
      } finally {
        await act(async () => { finish(refreshed); await pending })
      }
      expect(screen.getByLabelText(/^server$/i)).toHaveValue("docs-second")
    })

    it("preserves the import result when the imported server becomes selected", async () => {
      const rows = await mocks.listExternalServers()
      mocks.listExternalServers.mockResolvedValueOnce(rows).mockResolvedValue([
        rows[0], { ...rows[1], server_source: "managed", runtime_executable: true }
      ])
      render(<ExternalServersTab />)
      const row = await findServerRow("Search Legacy")
      fireEvent.click(within(row).getByRole("button", { name: /import to mcp hub/i }))
      await waitFor(() => expect(screen.getByLabelText(/^server$/i)).toHaveValue("search-legacy"))
      expect(await screen.findByText(/^Legacy server imported$/)).toBeTruthy()
    })

    it("preserves the slot creation result when the new slot becomes selected", async () => {
      const rows = await mocks.listExternalServers()
      mocks.listExternalServers.mockResolvedValueOnce(rows).mockResolvedValue([
        { ...rows[0], credential_slots: [...rows[0].credential_slots, {
          ...rows[0].credential_slots[0], slot_name: "token_write", display_name: "Write token"
        }] }, ...rows.slice(1)
      ])
      render(<ExternalServersTab />)
      await findCredentialInput()
      fireEvent.click(screen.getByRole("button", { name: /^add slot$/i }))
      fireEvent.change(screen.getByLabelText(/^slot name$/i), { target: { value: "token_write" } })
      fireEvent.change(screen.getByLabelText(/^slot display name$/i), { target: { value: "Write token" } })
      fireEvent.click(screen.getByRole("button", { name: /^save slot$/i }))
      await waitFor(() => expect(screen.getByLabelText(/^credential slot$/i)).toHaveValue("token_write"))
      expect(await screen.findByText(/^Credential slot created$/)).toBeTruthy()
    })

    it("preserves the new server discovery warning when its server becomes selected", async () => {
      const rows = await mocks.listExternalServers()
      const created = { ...rows[0], id: "docs-new", name: "Docs New", credential_slots: [] }
      mocks.listExternalServers.mockResolvedValueOnce(rows).mockResolvedValue([...rows, created])
      mocks.createExternalServer.mockResolvedValueOnce(created)
      mocks.refreshExternalServerDiscovery.mockRejectedValueOnce(new Error("discovery-unavailable"))
      render(<ExternalServersTab />)
      await findCredentialInput()
      fireEvent.click(screen.getByRole("button", { name: /new managed server/i }))
      fireEvent.click(screen.getByRole("button", { name: /local stdio/i }))
      fireEvent.change(screen.getByLabelText(/^server id$/i), { target: { value: "docs-new" } })
      fireEvent.change(screen.getByLabelText(/^name$/i), { target: { value: "Docs New" } })
      fireEvent.change(screen.getByLabelText(/^command$/i), { target: { value: "node" } })
      fireEvent.click(screen.getByRole("button", { name: /save and discover tools/i }))
      await waitFor(() => expect(screen.getByLabelText(/^server$/i)).toHaveValue("docs-new"))
      expect(await screen.findByText("Docs New saved, but tool discovery failed: discovery-unavailable")).toBeTruthy()
    })

    it("shows the current slot save configuration error without losing the draft", async () => {
      await configureServers()
      const guidance = "Configure a valid BYOK_ENCRYPTION_KEY before saving MCP credentials"
      mocks.setExternalServerSlotSecret.mockRejectedValueOnce(new Error(guidance))
      render(<ExternalServersTab />)
      const input = await findCredentialInput()
      fireEvent.change(input, { target: { value: "first-target-draft" } })
      fireEvent.click(screen.getByRole("button", { name: /^save slot secret$/i }))
      expect(await screen.findByText(`Failed to save slot secret: ${guidance}`)).toBeTruthy()
      expect(input).toHaveValue("first-target-draft")
    })
  })
})
