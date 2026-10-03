// @vitest-environment jsdom
/**
 * Red-first reproduction for #3104 (UX G03, CS-03) under decision D1: when the
 * server is connected, a new chat is saved to the server by default. The
 * `it.fails` asserts the correct behaviour and passes only while the defect
 * reproduces. The fake server persists turns written either through
 * POST /api/v1/chats/{id}/messages or through /chat/completions with
 * `save_to_db: true`, so the assertion does not prescribe the mechanism.
 */
import {
  createFakeTldwServer,
  renderPlayground,
  resetPlaygroundHarness,
  sendFromComposer
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"

describe("Playground server-side saving (#3104)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  // CS-03 #3104 — a fresh send only creates a local Dexie history (normalChatMode.ts:644-656); useChatActions.ts:3957-3965
  // skips server-chat creation whenever a HistorySelection controller is mounted, and chatModePipeline.ts:716 forces save_to_db: false.
  it.fails("CS-03 (#3104): sending in a fresh chat while connected creates a server conversation holding the turn", async () => {
    const server = createFakeTldwServer()
    server.setDefaultReply("Saved reply")
    const view = await renderPlayground({ server })

    await sendFromComposer(view, "Keep this on the server")

    expect(
      [...server.chats.values()].map((chat) =>
        chat.messages.map((message) => `${message.role}: ${message.content}`)
      )
    ).toEqual([["user: Keep this on the server", "assistant: Saved reply"]])
  })
})
