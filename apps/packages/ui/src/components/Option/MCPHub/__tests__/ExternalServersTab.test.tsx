// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Modal } from "antd"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

const mocks = vi.hoisted(() => ({
  invalidateQueries: vi.fn(),
  listExternalServers: vi.fn(),
  setExternalServerSecret: vi.fn(),
  setExternalServerSlotSecret: vi.fn(),
  clearExternalServerSlotSecret: vi.fn(),
  getExternalServerAuthTemplate: vi.fn(),
  updateExternalServerAuthTemplate: vi.fn(),
  importExternalServer: vi.fn(),
  createExternalServer: vi.fn(),
  updateExternalServer: vi.fn(),
  deleteExternalServer: vi.fn(),
  refreshExternalServerDiscovery: vi.fn(),
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
  updateExternalServerAuthTemplate: (...args: unknown[]) => mocks.updateExternalServerAuthTemplate(...args),
  importExternalServer: (...args: unknown[]) => mocks.importExternalServer(...args),
  createExternalServer: (...args: unknown[]) => mocks.createExternalServer(...args),
  updateExternalServer: (...args: unknown[]) => mocks.updateExternalServer(...args),
  deleteExternalServer: (...args: unknown[]) => mocks.deleteExternalServer(...args),
  refreshExternalServerDiscovery: (...args: unknown[]) => mocks.refreshExternalServerDiscovery(...args),
  createExternalServerCredentialSlot: (...args: unknown[]) => mocks.createExternalServerCredentialSlot(...args),
  updateExternalServerCredentialSlot: (...args: unknown[]) => mocks.updateExternalServerCredentialSlot(...args),
  deleteExternalServerCredentialSlot: (...args: unknown[]) => mocks.deleteExternalServerCredentialSlot(...args)
}))

import { ExternalServersTab } from "../ExternalServersTab"

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
})
