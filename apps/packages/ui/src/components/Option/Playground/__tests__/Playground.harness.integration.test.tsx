// @vitest-environment jsdom
/**
 * Self-checks for the Playground integration harness (#3125, Stage 0 of #3101).
 * These must stay green: they prove the harness mounts the real Playground,
 * drives its real send path, and that the fake server supports both local and
 * native (server-owned) history turns, so red-first reproductions elsewhere
 * fail for their defect and not for a harness gap.
 */
import {
  completionMessages,
  createFakeTldwServer,
  getHarnessDb,
  HARNESS_WAIT,
  openSavedServerChat,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  sendFromComposer,
  SidebarServerChatButton,
  waitFor
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

describe("Playground integration harness", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("sends one message from a fresh chat as exactly one completion request containing it", async () => {
    const server = createFakeTldwServer()
    server.setDefaultReply("Self-check reply")
    const view = await renderPlayground({ server })

    await sendFromComposer(view, "Harness self-check")

    const completions = server.completionRequests()
    expect(completions).toHaveLength(1)
    expect(completionMessages(completions[0])).toEqual([
      { role: "user", content: "Harness self-check" }
    ])
    expect(completions[0].body).toMatchObject({ model: "local-uat-chat", stream: true })
    await waitFor(
      () => expect(screen.getByText("Self-check reply")).toBeInTheDocument(),
      HARNESS_WAIT
    )
    expect(server.unhandled).toEqual([])
  })

  it("keeps the local turn in the in-memory chat database", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })

    await sendFromComposer(view, "Remember me")

    const historyId = useStoreMessageOption.getState().historyId
    expect(historyId).toBeTruthy()
    const rows = await getHarnessDb()
      .messages.where("history_id")
      .equals(historyId as string)
      .toArray()
    // Both rows of a turn can share a timestamp, so compare without order.
    expect(rows.map((row) => `${row.role}: ${row.content}`).sort()).toEqual([
      "assistant: Harness reply",
      "user: Remember me"
    ])
  })

  it("continues a saved server chat through native history admission", async () => {
    const server = createFakeTldwServer()
    server.seedChat({
      id: "saved-chat",
      title: "Saved topic",
      turns: [
        { role: "user", content: "Saved question" },
        { role: "assistant", content: "Saved answer" }
      ]
    })
    const view = await renderPlayground({
      server,
      extras: <SidebarServerChatButton chatId="saved-chat" title="Saved topic" />
    })
    await openSavedServerChat(view, { title: "Saved topic", lastMessage: "Saved answer" })

    await sendFromComposer(view, "Follow-up question")

    expect(server.completionRequests().map(completionMessages)).toEqual([
      [
        { role: "user", content: "Saved question" },
        { role: "assistant", content: "Saved answer" },
        { role: "user", content: "Follow-up question" }
      ]
    ])
    expect(
      server.chats.get("saved-chat")?.messages.map((message) => [message.role, message.content])
    ).toEqual([
      ["user", "Saved question"],
      ["assistant", "Saved answer"],
      ["user", "Follow-up question"],
      ["assistant", "Harness reply"]
    ])
    expect(server.unhandled).toEqual([])
  })
})
