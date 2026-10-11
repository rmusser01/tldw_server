import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
const keep = vi.hoisted(() =>
  vi.fn(async () => ({ status: "kept", retained: false }))
)
vi.mock("@/services/history-turn-keep", async (original) => ({
  ...(await original<typeof import("@/services/history-turn-keep")>()),
  keepHistoryTurnRecovery: keep
}))
import { useKeepInterruptedHistoryTurns } from "../useKeepInterruptedHistoryTurns"
import type { HistorySelectionController } from "../useHistorySelection"
import type { HistoryTurnRecovery } from "@/db/dexie/types"

describe("explicit edited branch recovery", () => {
  it.each(["edit_resend", "regenerate", "roleplay_resume"].flatMap(action => ["complete", "interrupted", "stopped"].map(outcome => ({ action, outcome }))))(
    "$action / $outcome remains in review without automatic keep, dismissal or transcript changes",
    async ({ action, outcome }) => {
      keep.mockClear()
      const turn: HistoryTurnRecovery = {
        operation_id: "edit",
        ...(action === "roleplay_resume" ? {roleplay_resume:{} as HistoryTurnRecovery["roleplay_resume"]} : {action:action as HistoryTurnRecovery["action"]}),
        persistence: "server",
        outcome: outcome as HistoryTurnRecovery["outcome"],
        owner_key: "owner",
        conversation_id: "chat",
        input_id: "edited-user",
        assistant_id: "edited-answer",
        origin_view: {
          owner_key: "owner",
          conversation_id: "chat",
          view_session_id: "origin",
          cursor: { kind: "before_message", message_id: "original-user" },
          selection_revision: 1,
          interpretation: { kind: "parent_graph_v1" }
        },
        selection_digest: "selection",
        request_context_digest: "request",
        state: "generated_unsaved",
        input_text: "Edited question",
        input_images: [],
        result_text: "Answer",
        created_at: 1,
        admission: {
          version: 1,
          owner_key: "owner",
          conversation_id: "chat",
          input_message_id: "edited-user",
          input_message_revision: "1",
          selection_digest: "selection"
        },
        ...(outcome === "complete"
          ? { settled_message_id: "edited-answer" }
          : {})
      }
      const entry = {
        turn,
        scope: { profile_id: "profile", client_session_id: "client" }
      }
      const selection = {
        status: "ready",
        owner: { kind: "native" },
        recoveries: [entry],
        refreshRecovery: vi.fn(),
        dismissRecovery: vi.fn()
      } as unknown as HistorySelectionController
      const setMessages = vi.fn()
      const hook = renderHook(() =>
        useKeepInterruptedHistoryTurns({
          selection,
          messages: [
            {
              id: "edited-user",
              isBot: false,
              name: "You",
              message: "Edited question",
              sources: []
            }
          ],
          setMessages
        })
      )
      await act(async () => {
        await Promise.resolve()
      })
      expect(keep).not.toHaveBeenCalled()
      expect(setMessages).not.toHaveBeenCalled()
      expect(selection.dismissRecovery).not.toHaveBeenCalled()
      expect(hook.result.current(entry)).toBe(false)
      hook.unmount()
    }
  )
})
