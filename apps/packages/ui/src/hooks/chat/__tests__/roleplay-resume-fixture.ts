import { resolveHistorySelection } from "@/utils/history-selection"
import type { HistoryCaptureRequestV1, HistoryCaptureResultV1 } from "@/types/history-selection"
import type { RoleplayResumeUserAppendRequest, RoleplayResumeCompletionRequest } from "@/services/tldw/roleplay-resume"

// Declared server responses; the actual sender, parsers and journal remain under test.
export const roleplayDetail = (id: string, tailId: string | null) => ({
  id, scope_type: "global", workspace_id: null,
  behavior_snapshot: { status: "valid", schema_version: 1, digest: "sha256:" + "c".repeat(64) },
  resume_eligible: true, resume_ineligible_reason: null,
  settings_version: 4, history_version: 9,
  tail: { message_id: tailId, message_version: tailId === null ? null : 1 }
})

export const roleplayAppend = (id: string, input: RoleplayResumeUserAppendRequest) => ({
  version: 1 as const, conversation_id: id, message_id: input.message_id, message_version: 1,
  parent_message_id: input.expected.tail.message_id,
  fence: { ...input.expected, history_version: input.expected.history_version + 1,
    tail: { message_id: input.message_id, message_version: 1 } }
})

export const roleplayCompletion = (id: string, body: RoleplayResumeCompletionRequest) => {
  const request = body.tldw_roleplay_resume_v1
  return {
    chat_id: id, character_id: 12, provider: "openai", model: "original-model", saved: true,
    user_message_id: request.input_message_id, assistant_message_id: request.assistant_message_id,
    assistant_content: "Original-roleplay reply",
    tldw_roleplay_resume_v1: { ...request, assistant_message_version: 1,
      fence: { ...request.expected, history_version: request.expected.history_version + 1,
        tail: { message_id: request.assistant_message_id, message_version: 1 } }, unsaved_reason: null }
  }
}

export type RoleplayFixtureRow = { id: string; revision: string; parent_id: string | null; role: "user" | "assistant"; settled: boolean; message: string }
export const captureRoleplayFixtureRows = (
  capture: HistoryCaptureResultV1,
  request: HistoryCaptureRequestV1,
  rows: RoleplayFixtureRow[],
  content: Map<string, import("@/types/history-selection").HistorySelectionCaptureV1["selected_content"][number]>
): HistoryCaptureResultV1 => {
  if (capture.status !== "captured") return capture
  for (const row of [...capture.selected_content, ...rows.map(row => ({ id: row.id, revision: row.revision, message: row.message, images: [] }))]) content.set(JSON.stringify([row.id, row.revision]), row)
  if (!rows.length) return capture
  const snapshot = { ...capture.snapshot,
    nodes: [...capture.snapshot.nodes, ...rows.map(({ message: _text, ...row }) => row)] }
  const selected = resolveHistorySelection(snapshot, capture.view, request.purpose, "")
  if (selected.status !== "ready") throw new Error(selected.code)
  return { ...capture, snapshot, rows: selected.rows,
    selected_content: selected.rows.map(row => {
      const value = content.get(JSON.stringify([row.id, row.revision]))
      if (!value) throw new Error("fixture_canonical_content_missing")
      return value
    }) }
}
