import { describe, it, expect, vi } from "vitest"
import {
  hydrateWorkspaceFromServer,
  optimisticWorkspaceUpdate,
} from "../workspace-api"

describe("workspace store API-first mutations", () => {
  it("hydrates workspace state from server on workspace switch", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      id: "ws-1",
      name: "Server WS",
      sources: [{ id: "src-1", title: "Video", version: 1 }],
      artifacts: [],
      notes: [],
      version: 3,
    })
    const state = await hydrateWorkspaceFromServer("ws-1", { fetch: mockFetch })
    expect(state.name).toBe("Server WS")
    expect(state.sources).toHaveLength(1)
    expect(state.version).toBe(3)
    expect(mockFetch).toHaveBeenCalledWith("ws-1")
  })

  it("hydrates with empty arrays when server returns no sub-resources", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      id: "ws-2",
      name: "Empty WS",
      version: 1,
    })
    const state = await hydrateWorkspaceFromServer("ws-2", { fetch: mockFetch })
    expect(state.sources).toEqual([])
    expect(state.artifacts).toEqual([])
    expect(state.notes).toEqual([])
  })

  it("maps backend workspace source and artifact fields into local workspace state", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      id: "ws-1",
      name: "Server WS",
      version: 2,
      sources: [
        {
          id: "src-1",
          workspace_id: "ws-1",
          media_id: 42,
          title: "Quarterly strategy doc",
          source_type: "document",
          url: "https://example.test/doc",
          selected: true,
          added_at: "2026-05-06T12:00:00Z",
          version: 1
        }
      ],
      artifacts: [
        {
          id: "art-1",
          workspace_id: "ws-1",
          artifact_type: "report",
          title: "Executive Brief",
          status: "draft",
          content: "Brief body",
          total_tokens: 120,
          total_cost_usd: 0.02,
          created_at: "2026-05-06T12:05:00Z",
          completed_at: "2026-05-06T12:06:00Z",
          version: 3
        }
      ],
      notes: []
    })

    const state = await hydrateWorkspaceFromServer("ws-1", { fetch: mockFetch })

    expect(state.sources[0]).toMatchObject({
      id: "src-1",
      mediaId: 42,
      title: "Quarterly strategy doc",
      type: "document",
      status: "ready"
    })
    expect(state.artifacts[0]).toMatchObject({
      id: "art-1",
      type: "report",
      title: "Executive Brief",
      status: "completed",
      reviewStatus: "draft",
      content: "Brief body",
      totalTokens: 120,
      totalCostUsd: 0.02
    })
  })

  it.each([
    "draft",
    "reviewing",
    "accepted",
    "needs_revision",
    "rejected",
    "exported",
    "assigned",
    "archived"
  ])(
    "preserves server review status %s while deriving completed generation state",
    async (reviewStatus) => {
      const mockFetch = vi.fn().mockResolvedValue({
        id: `ws-${reviewStatus}`,
        name: "Review WS",
        version: 1,
        sources: [],
        artifacts: [
          {
            id: `art-${reviewStatus}`,
            workspace_id: `ws-${reviewStatus}`,
            artifact_type: "report",
            title: "Review Artifact",
            status: reviewStatus,
            content: "Completed artifact body",
            total_tokens: null,
            total_cost_usd: null,
            created_at: "2026-05-06T12:05:00Z",
            completed_at: null,
            version: 1
          }
        ],
        notes: []
      })

      const state = await hydrateWorkspaceFromServer(`ws-${reviewStatus}`, {
        fetch: mockFetch
      })

      expect(state.artifacts[0]).toMatchObject({
        id: `art-${reviewStatus}`,
        status: "completed",
        reviewStatus
      })
    }
  )

  it("maps traceable artifact contract fields into generated artifacts", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      id: "ws-traceable",
      name: "Traceable WS",
      version: 4,
      sources: [],
      artifacts: [
        {
          id: "art-traceable",
          workspace_id: "ws-traceable",
          artifact_type: "report",
          title: "Reviewed ACP Brief",
          status: "completed",
          review_state: "accepted",
          content_type: "text/markdown",
          content: "# Brief\n\nAccepted body",
          preview_text: "Accepted body",
          summary: "A reviewed ACP brief",
          total_tokens: 560,
          total_cost_usd: 0.12,
          owner_scope: "workspace",
          owner_id: "workspace-owner",
          project_id: "project-9",
          task_id: "task-workspace-4",
          source_collection_id: "collection-3",
          root_artifact_id: "root-art-1",
          artifact_version_id: "version-art-3",
          previous_version_id: "version-art-2",
          schema_version: 1,
          producer_metadata: {
            producer_type: "acp",
            producer_id: "task-42",
            run_id: "run-7",
            session_id: "session-abc",
            links: {
              diagnostics: "/api/v1/acp/sessions/session-abc/diagnostics"
            }
          },
          source_lineage: {
            sources: [
              {
                source_id: "src-1",
                source_type: "media",
                label: "Transcript",
                media_id: 42,
                citation_spans: [{ start: 12, end: 48 }]
              }
            ]
          },
          review_metadata: {
            reviewer_id: "reviewer-1",
            decision: "accepted"
          },
          version_metadata: {
            revision_reason: "Reviewer accepted the brief"
          },
          export_refs: [{ format: "md", file_id: 101, status: "ready" }],
          redaction: {
            support_safe: true,
            redacted: false,
            retention_class: "standard"
          },
          created_at: "2026-05-06T12:05:00Z",
          completed_at: "2026-05-06T12:06:00Z",
          version: 3
        }
      ],
      notes: []
    })

    const state = await hydrateWorkspaceFromServer("ws-traceable", {
      fetch: mockFetch
    })

    expect(state.artifacts[0]).toMatchObject({
      id: "art-traceable",
      status: "completed",
      reviewStatus: "accepted",
      contentType: "text/markdown",
      previewText: "Accepted body",
      summary: "A reviewed ACP brief",
      ownerScope: "workspace",
      ownerId: "workspace-owner",
      projectId: "project-9",
      taskId: "task-workspace-4",
      sourceCollectionId: "collection-3",
      rootArtifactId: "root-art-1",
      artifactVersionId: "version-art-3",
      previousVersionId: "version-art-2",
      schemaVersion: 1,
      producerMetadata: {
        producerType: "acp",
        producerId: "task-42",
        runId: "run-7",
        sessionId: "session-abc"
      },
      sourceLineage: [
        {
          sourceId: "src-1",
          sourceType: "media",
          title: "Transcript",
          mediaId: 42,
          citationCount: 1
        }
      ],
      reviewMetadata: {
        reviewerId: "reviewer-1",
        decision: "accepted"
      },
      versionMetadata: {
        revisionReason: "Reviewer accepted the brief"
      },
      exportRefs: [{ format: "markdown", fileId: 101, status: "ready" }],
      exportTargets: ["markdown"],
      redaction: {
        supportSafe: true,
        redacted: false,
        retentionClass: "standard"
      }
    })
  })

  it("does not mark unknown backend artifact statuses as completed", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      id: "ws-unknown-status",
      name: "Server WS",
      version: 1,
      sources: [],
      artifacts: [
        {
          id: "art-queued",
          workspace_id: "ws-unknown-status",
          artifact_type: "report",
          title: "Queued Brief",
          status: "queued",
          content: null,
          total_tokens: null,
          total_cost_usd: null,
          created_at: "2026-05-06T12:05:00Z",
          completed_at: null,
          version: 1
        }
      ],
      notes: []
    })

    const state = await hydrateWorkspaceFromServer("ws-unknown-status", {
      fetch: mockFetch
    })

    expect(state.artifacts[0]).toMatchObject({
      id: "art-queued",
      status: "pending",
      reviewStatus: undefined
    })
  })

  it("performs optimistic update with rollback on 409", async () => {
    const mockUpdate = vi.fn().mockRejectedValue({
      status: 409,
      body: { version: 5, name: "Server Name" },
    })
    const result = await optimisticWorkspaceUpdate(
      { id: "ws-1", name: "Local Name", version: 3 },
      { name: "New Name" },
      { update: mockUpdate }
    )
    expect(result.name).toBe("Server Name")
    expect(result.version).toBe(5)
  })

  it("updates local store on successful server mutation", async () => {
    const mockUpdate = vi.fn().mockResolvedValue({
      id: "ws-1",
      name: "New",
      version: 4,
    })
    const result = await optimisticWorkspaceUpdate(
      { id: "ws-1", name: "Old", version: 3 },
      { name: "New" },
      { update: mockUpdate }
    )
    expect(result.name).toBe("New")
    expect(result.version).toBe(4)
  })

  it("rethrows non-409 errors", async () => {
    const mockUpdate = vi.fn().mockRejectedValue(new Error("Network error"))
    await expect(
      optimisticWorkspaceUpdate(
        { id: "ws-1", name: "X", version: 1 },
        { name: "Y" },
        { update: mockUpdate }
      )
    ).rejects.toThrow("Network error")
  })
})
