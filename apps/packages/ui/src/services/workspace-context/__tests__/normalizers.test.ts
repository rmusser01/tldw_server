import { describe, expect, it } from "vitest"
import type {
  WorkspaceApiResponse,
  WorkspaceCapabilitiesResponse,
  WorkspaceContextResponse,
  WorkspaceProjectRoot,
  WorkspaceSourceStatusSummary
} from "@/services/tldw/domains/workspace-api"
import {
  compareACPWorkspaceContext,
  normalizeActiveWorkspaceContext,
  normalizeWorkspaceSummary,
  resolveWorkspaceActionEligibility,
  resolveWorkspaceRecovery
} from "../normalizers"

const workspaceFixture = (
  overrides: Partial<WorkspaceApiResponse> = {}
): WorkspaceApiResponse => ({
  id: "ws-1",
  name: "Server Workspace",
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
  created_at: "2026-06-18T00:00:00Z",
  last_modified: "2026-06-18T00:10:00Z",
  version: 7,
  ...overrides
})

const sourceSummaryFixture = (
  overrides: Partial<WorkspaceSourceStatusSummary> = {}
): WorkspaceSourceStatusSummary => ({
  total: 0,
  selected: 0,
  queryable: 0,
  partially_queryable: 0,
  processing: 0,
  failed: 0,
  missing: 0,
  ...overrides
})

const projectRootFixture = (
  overrides: Partial<WorkspaceProjectRoot> = {}
): WorkspaceProjectRoot => ({
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
  mcp_trust_state: null,
  ...overrides
})

const capabilitiesFixture = (
  overrides: Partial<WorkspaceCapabilitiesResponse> = {}
): WorkspaceCapabilitiesResponse => ({
  workspace_id: "ws-1",
  workspace_profile: "research",
  workspace_kind: "research_workspace",
  access_level: "owner",
  source_summary: sourceSummaryFixture(),
  workspace_services: {},
  allowed_actions: {},
  ...overrides
})

const contextFixture = (
  overrides: Partial<WorkspaceContextResponse> = {}
): WorkspaceContextResponse => {
  const workspace = overrides.workspace ?? workspaceFixture()
  return {
    workspace_id: workspace.id,
    workspace_profile: workspace.workspace_profile,
    workspace_kind:
      workspace.workspace_profile === "project"
        ? "project_workspace"
        : "research_workspace",
    schema_version: 2,
    generated_at: "2026-06-18T00:11:00Z",
    workspace,
    attention_state: workspace.archived ? "archived" : "ready",
    resolution: { status: "complete", partial_errors: [] },
    project_root: projectRootFixture(),
    sources: {
      items: [],
      summary: sourceSummaryFixture()
    },
    capabilities: capabilitiesFixture({
      workspace_id: workspace.id,
      workspace_profile: workspace.workspace_profile
    }),
    services: {},
    allowed_actions: {},
    active_jobs: [],
    active_operations: [],
    partial_errors: [],
    ...overrides
  }
}

describe("workspace context normalizers", () => {
  it("keeps server workspace identity authoritative", () => {
    const summary = normalizeWorkspaceSummary(
      workspaceFixture({
        id: "ws-server-1",
        name: null,
        workspace_profile: "project"
      })
    )

    expect(summary.id).toBe("ws-server-1")
    expect(summary.label).toBe("Workspace ws-server-1")
    expect(summary.profile).toBe("project")
    expect(summary.version).toBe(7)
  })

  it("maps partial workspace context to visible recovery copy", () => {
    const context = normalizeActiveWorkspaceContext(
      contextFixture({
        attention_state: "needs_attention",
        resolution: {
          status: "partial",
          partial_errors: [
            {
              scope: "sources",
              code: "source_status_unavailable",
              message: "Source status unavailable"
            }
          ]
        },
        sources: {
          items: [],
          summary: sourceSummaryFixture({ total: 2 })
        },
        partial_errors: [
          {
            scope: "sources",
            code: "source_status_unavailable",
            message: "Source status unavailable"
          }
        ]
      })
    )

    expect(context.state).toBe("partial")
    expect(context.workspace?.label).toBe("Server Workspace")
    expect(context.sourceSummary.total).toBe(2)
    expect(context.recovery.reasonCode).toBe("partial_context")
    expect(context.recovery.message).toMatch(/partially resolved/i)
  })

  it("uses server action reason codes for eligibility recovery", () => {
    const decision = resolveWorkspaceActionEligibility("open_terminal", {
      allowed: false,
      reason_code: "workspace_project_root_missing"
    })

    expect(decision.allowed).toBe(false)
    expect(decision.reasonCode).toBe("workspace_project_root_missing")
    expect(decision.recovery.nextStepHref).toBe("#/workspaces")
    expect(decision.recovery.nextStepLabel).toMatch(/Workspaces/i)
  })

  it("allows action when the server allowed action permits it", () => {
    const decision = resolveWorkspaceActionEligibility("read_sources", {
      allowed: true,
      reason_code: null
    })

    expect(decision.allowed).toBe(true)
    expect(decision.recovery.reasonCode).toBe("allowed")
  })

  it("returns missing context when no server workspace id exists", () => {
    const context = normalizeActiveWorkspaceContext(null)

    expect(context.state).toBe("none")
    expect(context.workspaceId).toBeNull()
    expect(context.recovery.reasonCode).toBe("no_active_workspace")
  })

  it("detects ACP session and active workspace mismatch without mutating either side", () => {
    const context = compareACPWorkspaceContext({
      sessionWorkspaceId: "ws-session",
      activeWorkspaceId: "ws-active"
    })

    expect(context.state).toBe("mismatch")
    expect(context.sessionWorkspaceId).toBe("ws-session")
    expect(context.activeWorkspaceId).toBe("ws-active")
    expect(context.recovery.reasonCode).toBe("workspace_mismatch")
  })

  it("reports ACP alignment when session and active workspace match", () => {
    const context = compareACPWorkspaceContext({
      sessionWorkspaceId: "ws-1",
      activeWorkspaceId: "ws-1",
      activeWorkspaceLabel: "Server Workspace"
    })

    expect(context.state).toBe("aligned")
    expect(context.message).toMatch(/Server Workspace/)
    expect(context.recovery.reasonCode).toBe("aligned")
  })

  it("provides stable copy for unknown reason codes", () => {
    const recovery = resolveWorkspaceRecovery("future_reason_code")

    expect(recovery.reasonCode).toBe("future_reason_code")
    expect(recovery.message).toMatch(/cannot complete/i)
    expect(recovery.nextStepHref).toBe("#/workspaces")
  })
})
