import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { WorkspaceTrustPanel } from "../WorkspaceTrustPanel"
import type {
  WorkspaceCapabilitiesResponse,
  WorkspaceSourceStatusListResponse
} from "@/services/tldw/domains/workspace-api"

const statusPayload: WorkspaceSourceStatusListResponse = {
  workspace_id: "ws-1",
  sources: [
    {
      id: "src-ready",
      workspace_id: "ws-1",
      media_id: 1,
      title: "Ready source",
      source_type: "pdf",
      selected: true,
      state: "queryable",
      status_reason: "source_queryable",
      readiness: {
        metadata_ready: true,
        text_extracted: true,
        fts_ready: true,
        vector_ready: true,
        citation_ready: true,
        summary_ready: false,
        tool_accessible: true
      },
      progress_percent: 100,
      progress_message: "Ready for grounded questions.",
      job: null,
      updated_at: "2026-05-23T12:00:00Z"
    }
  ],
  summary: {
    total: 3,
    selected: 2,
    queryable: 1,
    partially_queryable: 1,
    processing: 1,
    failed: 0,
    missing: 1
  }
}

const capabilitiesPayload: WorkspaceCapabilitiesResponse = {
  workspace_id: "ws-1",
  workspace_kind: "research_workspace",
  access_level: "owner",
  source_summary: statusPayload.summary,
  workspace_services: {
    migration: {
      state: "available",
      reason_code: null,
      management_surface: "research_workspace_import"
    },
    sharing: {
      state: "private",
      reason_code: null,
      management_surface: "shared_workspaces"
    },
    mcp: {
      state: "not_configured",
      reason_code: "no_workspace_mcp_binding",
      management_surface: "mcp_hub"
    },
    acp: {
      state: "not_configured",
      reason_code: "no_workspace_acp_binding",
      management_surface: "acp_workspace"
    },
    sandbox: {
      state: "not_configured",
      reason_code: "no_workspace_sandbox_binding",
      management_surface: "sandbox_settings"
    },
    provider: {
      state: "unknown",
      reason_code: "provider_not_evaluated",
      management_surface: "model_settings"
    }
  },
  allowed_actions: {
    ask_grounded_questions: {
      allowed: false,
      reason_code: "no_queryable_sources"
    },
    run_mcp_tools: {
      allowed: false,
      reason_code: "mcp_not_configured"
    }
  }
}

describe("WorkspaceTrustPanel", () => {
  it("shows a neutral readiness check before backend trust data arrives", () => {
    render(<WorkspaceTrustPanel sourceStatus={null} capabilities={null} />)

    expect(screen.getByTestId("workspace-trust-panel")).toBeInTheDocument()
    expect(screen.getByText("Workspace trust")).toBeInTheDocument()
    expect(screen.getByText("Checking workspace readiness")).toBeInTheDocument()
    expect(screen.queryByText("0 queryable")).not.toBeInTheDocument()
    expect(screen.queryByText("0 processing")).not.toBeInTheDocument()
    expect(screen.queryByText("0 missing")).not.toBeInTheDocument()
  })

  it("renders source readiness summary and capability reasons", () => {
    render(
      <WorkspaceTrustPanel
        sourceStatus={statusPayload}
        capabilities={capabilitiesPayload}
      />
    )

    expect(screen.getByTestId("workspace-trust-panel")).toBeInTheDocument()
    expect(screen.getByText("Workspace trust")).toBeInTheDocument()
    expect(screen.getByText("1 queryable")).toBeInTheDocument()
    expect(screen.getByText("1 processing")).toBeInTheDocument()
    expect(screen.getByText("1 missing")).toBeInTheDocument()
    expect(screen.getByText("Grounded questions blocked")).toBeInTheDocument()
    expect(screen.getByText("no_queryable_sources")).toBeInTheDocument()
    expect(screen.getByText("MCP Hub")).toBeInTheDocument()
    expect(screen.getByText("no_workspace_mcp_binding")).toBeInTheDocument()
    expect(screen.getByText("ACP")).toBeInTheDocument()
    expect(screen.getByText("Sandbox")).toBeInTheDocument()
    expect(screen.getByText("Provider")).toBeInTheDocument()
  })

  it("renders a bounded warning when trust data cannot be loaded", () => {
    render(
      <WorkspaceTrustPanel
        sourceStatus={null}
        capabilities={null}
        errorMessage="Status API unavailable"
      />
    )

    expect(screen.getByRole("status")).toHaveTextContent("Status API unavailable")
    expect(screen.getByText("Workspace trust unavailable")).toBeInTheDocument()
  })
})
