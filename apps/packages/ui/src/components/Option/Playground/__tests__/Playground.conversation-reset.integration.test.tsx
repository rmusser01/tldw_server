// @vitest-environment jsdom
/**
 * Red-first reproductions for #3106 (UX G05): "New chat" and "Clear
 * conversation" must start a clean conversation. Each `it.fails` asserts the
 * correct behaviour and passes only while the defect reproduces; once a fix
 * lands it starts failing, prompting conversion to a plain `it`.
 */
import {
  clickHeaderNewChat,
  completionMessages,
  createFakeTldwServer,
  getHarnessDb,
  HeaderNewChatButton,
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

/** A local conversation's rows as sorted "role: content" lines (a turn's rows can share a timestamp). */
const localMessages = async (historyId: string) =>
  (await getHarnessDb().messages.where("history_id").equals(historyId).toArray())
    .map((row) => `${row.role}: ${row.content}`)
    .sort()

describe("Playground conversation reset (#3106)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  // CS-01 #3106 — useClearChat.ts:93-150 resets messages/history/historyId/serverChatId but never the
  // HistorySelection controller; useChatActions.ts:3450-3455 and normalChatMode.ts:642-643 then reuse the stale owner.
  it.fails("CS-01 (#3106): after New chat the next request carries no prior messages and is not appended to the previous local chat", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server, extras: <HeaderNewChatButton /> })
    await sendFromComposer(view, "First topic question")
    const previousHistoryId = useStoreMessageOption.getState().historyId as string
    expect(await localMessages(previousHistoryId)).toHaveLength(2)

    await clickHeaderNewChat(view)
    await sendFromComposer(view, "Unrelated question")

    expect(completionMessages(server.completionRequests()[1])).toEqual([
      { role: "user", content: "Unrelated question" }
    ])
    expect(await localMessages(previousHistoryId)).toEqual([
      "assistant: Harness reply",
      "user: First topic question"
    ])
    expect(screen.queryByText("First topic question")).not.toBeInTheDocument()
  })

  // CS-01 #3106 — same root cause: after a saved server chat was opened, the native owner survives New chat
  // (useClearChat.ts:93-150), so the next send is admitted into the old conversation (normalChatMode.ts:642-643).
  it.fails("CS-01 (#3106): after New chat from a saved server chat the next send posts nothing to that chat and carries no prior messages", async () => {
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
      extras: (
        <>
          <HeaderNewChatButton />
          <SidebarServerChatButton chatId="saved-chat" title="Saved topic" />
        </>
      )
    })
    await openSavedServerChat(view, { title: "Saved topic", lastMessage: "Saved answer" })

    await clickHeaderNewChat(view)
    await sendFromComposer(view, "Unrelated question")

    expect(server.find("POST", "/api/v1/chats/saved-chat/messages")).toEqual([])
    expect(server.completionRequests().map(completionMessages)).toEqual([
      [{ role: "user", content: "Unrelated question" }]
    ])
    expect(server.chats.get("saved-chat")?.messages).toHaveLength(2)
  })

  // CS-05 #3106 — PlaygroundForm.tsx:3468-3497 handleClearContext only calls setHistory([]) (L3487); messages and
  // the HistorySelection cursor stay, and normalChatMode.ts:848-886 rebuilds the request from that selection.
  it.fails("CS-05 (#3106): Clear conversation removes the messages and the next request carries no prior messages", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    await sendFromComposer(view, "Sensitive first question")

    await view.user.click(screen.getByRole("button", { name: "More tools" }))
    await view.user.click(await screen.findByText("Clear conversation"))
    await view.user.click(await screen.findByRole("button", { name: "Confirm" }))

    await waitFor(
      () => expect(screen.queryByText("Sensitive first question")).not.toBeInTheDocument(),
      { timeout: 3_000 }
    )
    await sendFromComposer(view, "Next question")
    expect(completionMessages(server.completionRequests()[1])).toEqual([
      { role: "user", content: "Next question" }
    ])
  })
})
