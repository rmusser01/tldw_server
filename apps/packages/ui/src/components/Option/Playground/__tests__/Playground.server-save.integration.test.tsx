// @vitest-environment jsdom
/**
 * #3104 (UX G03, CS-03) under decision D1: when the server is connected, a new
 * chat is saved to the server by default. This began as a red-first `it.fails`
 * reproduction; dev's #3195 (native ownership before a fresh saved send) makes
 * it hold, so it is a plain test now. The fake server persists turns written either through
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

  // CS-03 #3104 (fixed by #3195) — a fresh send used to create only a local Dexie history, because useChatActions skipped
  // server-chat creation whenever a HistorySelection controller was mounted and the pipeline forced save_to_db: false.
  it("CS-03 (#3104): sending in a fresh chat while connected creates a server conversation holding the turn", async () => {
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
