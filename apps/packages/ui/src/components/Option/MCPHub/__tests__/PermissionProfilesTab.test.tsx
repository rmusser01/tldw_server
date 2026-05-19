// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { Modal } from "antd"

const mocks = vi.hoisted(() => ({
  listPermissionProfiles: vi.fn(),
  createPermissionProfile: vi.fn(),
  updatePermissionProfile: vi.fn(),
  deletePermissionProfile: vi.fn(),
  getToolRegistrySummary: vi.fn(),
  listPathScopeObjects: vi.fn(),
  listExternalServers: vi.fn(),
  listProfileCredentialBindings: vi.fn(),
  upsertProfileCredentialBinding: vi.fn(),
  deleteProfileCredentialBinding: vi.fn()
}))

vi.mock("@/services/tldw/mcp-hub", () => ({
  listPermissionProfiles: (...args: unknown[]) => mocks.listPermissionProfiles(...args),
  createPermissionProfile: (...args: unknown[]) => mocks.createPermissionProfile(...args),
  updatePermissionProfile: (...args: unknown[]) => mocks.updatePermissionProfile(...args),
  deletePermissionProfile: (...args: unknown[]) => mocks.deletePermissionProfile(...args),
  getToolRegistrySummary: (...args: unknown[]) => mocks.getToolRegistrySummary(...args),
  listPathScopeObjects: (...args: unknown[]) => mocks.listPathScopeObjects(...args),
  listExternalServers: (...args: unknown[]) => mocks.listExternalServers(...args),
  listProfileCredentialBindings: (...args: unknown[]) => mocks.listProfileCredentialBindings(...args),
  upsertProfileCredentialBinding: (...args: unknown[]) => mocks.upsertProfileCredentialBinding(...args),
  deleteProfileCredentialBinding: (...args: unknown[]) => mocks.deleteProfileCredentialBinding(...args)
}))

import { PermissionProfilesTab } from "../PermissionProfilesTab"

describe("PermissionProfilesTab", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.listPermissionProfiles.mockResolvedValue([
      {
        id: 5,
        name: "Process Exec",
        owner_scope_type: "user",
        owner_scope_id: 7,
        mode: "custom",
        policy_document: {
          capabilities: ["process.execute"],
          allowed_tools: ["Bash(git *)"]
        },
        is_active: true
      }
    ])
    mocks.createPermissionProfile.mockResolvedValue({
      id: 6,
      name: "Read Only",
      owner_scope_type: "global",
      owner_scope_id: null,
      mode: "preset",
      policy_document: {
        capabilities: ["filesystem.read"]
      },
      is_active: true
    })
    mocks.listPathScopeObjects.mockResolvedValue([
      {
        id: 41,
        name: "Docs Only",
        owner_scope_type: "global",
        path_scope_document: {
          path_scope_mode: "workspace_root",
          path_allowlist_prefixes: ["docs"]
        },
        is_active: true
      }
    ])
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
    mocks.listExternalServers.mockResolvedValue([
      {
        id: "docs-managed",
        name: "Docs Managed",
        enabled: true,
        owner_scope_type: "global",
        transport: "stdio",
        config: {},
        secret_configured: true,
        server_source: "managed",
        binding_count: 1,
        runtime_executable: true,
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
      },
      {
        id: "legacy-search",
        name: "Legacy Search",
        enabled: true,
        owner_scope_type: "global",
        transport: "websocket",
        config: {},
        secret_configured: false,
        server_source: "legacy",
        binding_count: 0,
        runtime_executable: false,
        credential_slots: []
      },
      {
        id: "search-api",
        name: "Search API",
        enabled: true,
        owner_scope_type: "global",
        transport: "websocket",
        config: {},
        secret_configured: false,
        server_source: "managed",
        binding_count: 0,
        runtime_executable: true,
        credential_slots: [
          {
            server_id: "search-api",
            slot_name: "token_write",
            display_name: "Write token",
            secret_kind: "bearer_token",
            privilege_class: "write",
            is_required: true,
            secret_configured: false
          }
        ]
      }
    ])
    mocks.listProfileCredentialBindings.mockResolvedValue([
      {
        id: 41,
        binding_target_type: "permission_profile",
        binding_target_id: "5",
        external_server_id: "docs-managed",
        slot_name: "token_readonly",
        credential_ref: "slot",
        binding_mode: "grant",
        usage_rules: {}
      }
    ])
    mocks.upsertProfileCredentialBinding.mockResolvedValue({
      id: 42,
      binding_target_type: "permission_profile",
      binding_target_id: "5",
      external_server_id: "search-api",
      slot_name: "token_write",
      credential_ref: "slot",
      binding_mode: "grant",
      usage_rules: {}
    })
  })

  it("renders saved permission profiles and opens the create form", async () => {
    const user = userEvent.setup()
    render(<PermissionProfilesTab />)

    expect(await screen.findByText("Process Exec")).toBeTruthy()
    expect(screen.getByText("process.execute")).toBeTruthy()

    await user.click(screen.getByRole("button", { name: /new profile/i }))
    expect(screen.getByLabelText(/profile name/i)).toBeTruthy()
    expect(screen.getByText(/path scope source/i)).toBeTruthy()
    expect(screen.getByText(/allowed modules and tools/i)).toBeTruthy()
    expect(screen.getByText(/no additional restrictions/i)).toBeTruthy()
    await user.click(screen.getByRole("checkbox", { name: /notes\.search/i }))
    expect(screen.getByText(/local file scope/i)).toBeTruthy()
    expect(screen.getByText(/workspace root/i)).toBeTruthy()
  })

  it("manages profile external server grants and excludes legacy servers", async () => {
    const user = userEvent.setup()
    render(<PermissionProfilesTab />)

    await screen.findByText("Process Exec")
    await user.click(screen.getByRole("button", { name: "Edit" }))

    expect(await screen.findByText("External Service Bindings")).toBeTruthy()
    expect(screen.getByRole("checkbox", { name: /Read-only token/ })).toBeTruthy()
    expect(screen.getByRole("checkbox", { name: /Write token/ })).toBeTruthy()
    expect(screen.queryByRole("checkbox", { name: /Legacy Search/ })).toBeNull()
    expect(screen.getByText(/slot secret missing/i)).toBeTruthy()

    await user.click(screen.getByRole("checkbox", { name: /Write token/ }))

    expect(mocks.upsertProfileCredentialBinding).toHaveBeenCalledWith(5, "search-api", "token_write")
  })

  it("deletes a profile when the user confirms the modal", async () => {
    mocks.deletePermissionProfile.mockResolvedValue(undefined)
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementation((config: any) => {
      void config?.onOk?.()
      return { destroy: vi.fn(), update: vi.fn() } as any
    })
    const user = userEvent.setup()
    render(<PermissionProfilesTab />)

    await screen.findByText("Process Exec")
    await user.click(screen.getByRole("button", { name: "Delete" }))

    expect(confirmSpy).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Delete Profile" })
    )
    expect(mocks.deletePermissionProfile).toHaveBeenCalledWith(5)
  })

  it("does not delete a profile when the user cancels the modal", async () => {
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementation(() => {
      return { destroy: vi.fn(), update: vi.fn() } as any
    })
    const user = userEvent.setup()
    render(<PermissionProfilesTab />)

    await screen.findByText("Process Exec")
    await user.click(screen.getByRole("button", { name: "Delete" }))

    expect(confirmSpy).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Delete Profile" })
    )
    expect(mocks.deletePermissionProfile).not.toHaveBeenCalled()
  })
})
