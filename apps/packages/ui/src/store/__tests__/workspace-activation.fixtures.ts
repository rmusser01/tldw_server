import type { ServerWorkspaceState } from "../workspace-api"
import type { WorkspaceApiResponse } from "@/services/tldw/domains/workspace-api"

// Unit regression data only; this is not a live API or browser UAT fixture.
export const serverWorkspaceMetadata: WorkspaceApiResponse = {
  id: "server-research", name: "Server Research", archived: false,
  deleted: false, workspace_profile: "research", study_materials_policy: "general",
  banner_title: "Research Banner", banner_subtitle: "Server subtitle", banner_color: "green",
  audio_provider: "openai", audio_model: "tts-1", audio_voice: "alloy", audio_speed: 1.25,
  created_at: "2026-09-01T00:00:00Z", last_modified: "2026-09-02T00:00:00Z", version: 4,
  assistant_defaults: { assistant_kind: "persona", assistant_id: "persona-1", persona_memory_mode: "read_only" }
}

export const serverWorkspacePayload = (): ServerWorkspaceState => ({
  ...serverWorkspaceMetadata,
  metadata: { ...serverWorkspaceMetadata },
  sources: [{
    id: "server-source", workspace_id: "server-research", media_id: 101,
    title: "Server Source", source_type: "document", url: null, position: 0,
    selected: false, added_at: "2026-09-01T00:00:00Z", version: 1
  }],
  artifacts: [{
    id: "server-artifact", workspace_id: "server-research", artifact_type: "report",
    title: "Server Report", status: "completed", content: "Report body",
    total_tokens: 10, total_cost_usd: 0.01, created_at: "2026-09-01T00:00:00Z",
    completed_at: "2026-09-01T00:00:00Z", version: 1
  }],
  notes: [{
    id: 7, workspace_id: "server-research", title: "Canonical Note", content: "Canonical body",
    keywords_json: '["canonical"]', created_at: "2026-09-01T00:00:00Z",
    last_modified: "2026-09-02T00:00:00Z", version: 2
  }]
})
