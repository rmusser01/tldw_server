import { describe, expect, it } from "vitest"
import {
  buildWorkspaceIndexPath,
  normalizeWorkspaceIndexResponse
} from "../normalizers"

describe("workspace index normalizers", () => {
  it("normalizes empty index payloads into stable display defaults", () => {
    const snapshot = normalizeWorkspaceIndexResponse({})

    expect(snapshot.schemaVersion).toBe(1)
    expect(snapshot.workspaceId).toBe("")
    expect(snapshot.workspace.id).toBe("")
    expect(snapshot.membershipSummary).toEqual({
      total: 0,
      byResourceType: {},
      byRole: {}
    })
    expect(snapshot.resourceGroups).toEqual([])
    expect(snapshot.runtimeSummary).toEqual({
      total: 0,
      byKind: {},
      byStatus: {},
      bindings: []
    })
    expect(snapshot.warnings).toEqual([])
    expect(snapshot.recentActivity).toEqual([])
  })

  it("preserves unknown warning reason codes and server owner hrefs", () => {
    const snapshot = normalizeWorkspaceIndexResponse({
      workspace_id: "workspace-1",
      schema_version: 1,
      generated_at: "2026-06-18T12:00:00Z",
      workspace: {
        id: "workspace-1",
        name: "Research Workspace",
        workspace_profile: "project",
        archived: false,
        deleted: false
      },
      membership_summary: {
        total: 1,
        by_resource_type: { chat: 1 },
        by_role: { conversation: 1 }
      },
      resource_groups: [
        {
          resource_type: "chat",
          count: 1,
          owner_surface: { label: "Chat", href: "#/chat" },
          items: [
            {
              workspace_id: "workspace-1",
              resource_type: "chat",
              resource_id: "chat-1",
              role: "conversation",
              transfer_policy: "link",
              provenance: {},
              metadata: {},
              summary: {
                title: "Planning chat",
                href: "#/chat/chat-1",
                state: "available"
              },
              created_at: "2026-06-18T12:00:00Z",
              updated_at: "2026-06-18T12:00:00Z",
              version: 1,
              deleted: false
            }
          ]
        }
      ],
      warnings: [
        {
          severity: "future-severity",
          reason_code: "future_agent_runtime_notice",
          message: "The server exposed a future warning.",
          resource_type: "acp_session",
          resource_id: "session-1",
          action_href: "#/agent-playground"
        }
      ],
      recent_activity: [
        {
          workspace_id: "workspace-1",
          event_id: "event-1",
          event_type: "membership.linked",
          category: "membership",
          resource_type: "chat",
          resource_id: "chat-1",
          metadata: { role: "conversation" },
          created_at: "2026-06-18T12:00:00Z",
          version: 1
        }
      ]
    })

    expect(snapshot.workspace.profile).toBe("project")
    expect(snapshot.resourceGroups[0].ownerSurface.href).toBe("#/chat")
    expect(snapshot.resourceGroups[0].items[0].summary?.href).toBe("#/chat/chat-1")
    expect(snapshot.warnings[0]).toMatchObject({
      severity: "warning",
      reasonCode: "future_agent_runtime_notice",
      actionHref: "#/agent-playground"
    })
    expect(snapshot.recentActivity[0]).toMatchObject({
      eventType: "membership.linked",
      metadata: { role: "conversation" }
    })
  })

  it("builds the server index endpoint path with bounded query parameters", () => {
    expect(buildWorkspaceIndexPath("workspace 1", { groupLimit: 3, activityLimit: 7 })).toBe(
      "/api/v1/workspaces/workspace%201/index?group_limit=3&activity_limit=7"
    )
    expect(buildWorkspaceIndexPath("workspace/1", { groupLimit: 0, activityLimit: 999 })).toBe(
      "/api/v1/workspaces/workspace%2F1/index?group_limit=1&activity_limit=100"
    )
  })
})
