import { describe, expect, it } from "vitest"
import type {
  WorkspaceApiResponse,
  WorkspaceContextResponse
} from "@/services/tldw/domains/workspace-api"
import {
  WORKSPACE_MANAGER_CANONICAL_LABELS,
  WORKSPACE_MANAGER_COPY,
  isCanonicalWorkspaceManagerLabel
} from "../workspace-manager-copy"
import { normalizeWorkspaceManagerItem } from "../workspace-manager-models"

const baseWorkspace = (
  overrides: Partial<WorkspaceApiResponse> = {}
): WorkspaceApiResponse => ({
  id: "ws-1",
  name: "Workspace One",
  archived: false,
  study_materials_policy: "workspace",
  workspace_profile: "research",
  deleted: false,
  banner_title: null,
  banner_subtitle: null,
  banner_color: null,
  audio_provider: null,
  audio_model: null,
  audio_voice: null,
  audio_speed: null,
  created_at: "2026-06-04T00:00:00Z",
  last_modified: "2026-06-04T00:00:00Z",
  version: 1,
  ...overrides
})

const contextFor = (
  workspace: WorkspaceApiResponse,
  overrides: Partial<WorkspaceContextResponse> = {}
): WorkspaceContextResponse => ({
  workspace_id: workspace.id,
  workspace_profile: workspace.workspace_profile,
  workspace_kind:
    workspace.workspace_profile === "project"
      ? "project_workspace"
      : "research_workspace",
  schema_version: 2,
  generated_at: "2026-06-04T00:00:01Z",
  workspace,
  attention_state: "ready",
  resolution: { status: "complete", partial_errors: [] },
  project_root: {
    state: "not_configured",
    root_id: null,
    backend: null,
    display_name: null,
    path_hint: null,
    git_state: null,
    file_inventory_state: "not_started",
    file_inventory: {
      state: "not_started",
      indexed_file_count: null,
      total_file_count: null,
      updated_at: null,
      available: false
    },
    indexing_state: null,
    sandbox_mount_state: null,
    mcp_trust_state: null
  },
  sources: {
    items: [],
    summary: {
      total: 0,
      selected: 0,
      queryable: 0,
      partially_queryable: 0,
      processing: 0,
      failed: 0,
      missing: 0
    }
  },
  capabilities: {} as WorkspaceContextResponse["capabilities"],
  services: {},
  allowed_actions: {},
  active_jobs: [],
  active_operations: [],
  partial_errors: [],
  ...overrides
})

describe("workspace manager model normalization", () => {
  it("uses context attention and project root inventory when context is present", () => {
    const workspace = baseWorkspace({
      workspace_profile: "project",
      name: "Project Workspace"
    })
    const item = normalizeWorkspaceManagerItem(
      workspace,
      contextFor(workspace, {
        attention_state: "working",
        project_root: {
          state: "attached",
          root_id: "root-1",
          backend: "host_local",
          display_name: "Repo",
          path_hint: "/repo",
          git_state: "clean",
          file_inventory_state: "current",
          file_inventory: {
            state: "current",
            indexed_file_count: 12,
            total_file_count: 14,
            updated_at: "2026-06-04T00:00:02Z",
            available: true
          },
          indexing_state: null,
          sandbox_mount_state: null,
          mcp_trust_state: "trusted"
        },
        active_operations: [
          {
            operation_id: "op-1",
            workspace_id: workspace.id,
            command: "provision_sandbox_root",
            status: "running",
            started_at: "2026-06-04T00:00:00Z",
            updated_at: "2026-06-04T00:00:01Z",
            retryable: true,
            diagnostics: {},
            poll_href: "/api/v1/workspaces/ws-1/operations/op-1"
          }
        ]
      })
    )

    expect(item.profile).toBe("project")
    expect(item.attentionState).toBe("working")
    expect(item.projectRoot?.backend).toBe("host_local")
    expect(item.projectRoot?.fileInventory.available).toBe(true)
    expect(item.activeOperations).toHaveLength(1)
  })

  it("defaults project workspaces without context or roots to setup pending", () => {
    const item = normalizeWorkspaceManagerItem(
      baseWorkspace({ workspace_profile: "project" })
    )

    expect(item.attentionState).toBe("setup_pending")
    expect(item.projectRoot?.state).toBe("not_configured")
    expect(item.projectRoot?.fileInventory.available).toBe(false)
  })

  it("defaults research workspaces without context to ready", () => {
    const item = normalizeWorkspaceManagerItem(baseWorkspace())

    expect(item.profile).toBe("research")
    expect(item.attentionState).toBe("ready")
  })

  it("normalizes malformed active operations to an empty list", () => {
    const workspace = baseWorkspace()
    const item = normalizeWorkspaceManagerItem(
      workspace,
      contextFor(workspace, {
        active_operations: "invalid" as unknown as WorkspaceContextResponse["active_operations"]
      })
    )

    expect(item.activeOperations).toEqual([])
  })

  it("normalizes archived and unknown workspace inputs conservatively", () => {
    expect(
      normalizeWorkspaceManagerItem(
        baseWorkspace({ archived: true, workspace_profile: "project" })
      ).attentionState
    ).toBe("archived")

    expect(
      normalizeWorkspaceManagerItem({
        ...baseWorkspace(),
        workspace_profile: "legacy" as WorkspaceApiResponse["workspace_profile"]
      }).attentionState
    ).toBe("needs_attention")
  })
})

describe("workspace manager copy guardrails", () => {
  it("pins canonical manager labels", () => {
    expect(WORKSPACE_MANAGER_COPY.workspace).toBe("Workspace")
    expect(WORKSPACE_MANAGER_COPY.researchWorkspace).toBe("Research Workspace")
    expect(WORKSPACE_MANAGER_COPY.projectWorkspace).toBe("Project Workspace")
    expect(WORKSPACE_MANAGER_COPY.hostLocalRoot).toBe("Host-local root")
    expect(WORKSPACE_MANAGER_COPY.sandboxManagedRoot).toBe("Sandbox-managed root")
    expect(WORKSPACE_MANAGER_COPY.mcpTrustedRootBinding).toBe(
      "MCP trusted root binding"
    )
    expect(WORKSPACE_MANAGER_COPY.mcpToolScope).toBe("MCP tool scope")
    expect(WORKSPACE_MANAGER_COPY.agentExecutionWorkspace).toBe(
      "agent execution workspace"
    )
  })

  it("rejects prototype and MCP labels as canonical manager labels", () => {
    expect(WORKSPACE_MANAGER_CANONICAL_LABELS).not.toContain(
      "Workspace Playground"
    )
    expect(WORKSPACE_MANAGER_CANONICAL_LABELS).not.toContain("Shared Workspace")
    expect(isCanonicalWorkspaceManagerLabel("Workspace Playground")).toBe(false)
    expect(isCanonicalWorkspaceManagerLabel("Shared Workspace")).toBe(false)
  })
})
