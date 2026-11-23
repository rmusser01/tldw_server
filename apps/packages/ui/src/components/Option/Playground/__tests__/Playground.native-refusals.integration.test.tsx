// @vitest-environment jsdom
/**
 * Server refusals of server-settled turns, end to end (D7 P5, #3101).
 *
 * The fake server answers with the real server's 409 bodies
 * (`history_edit_target_changed`, `history_branch_changed`), and the refusal
 * reaches the turn through the real bgStream and TldwChat, which wraps it as
 * "Stream completion failed" with the HTTP error as its cause. The turn must
 * still read it as a definitive refusal: nothing was saved, the user's input
 * is kept, and the page offers the matching recovery.
 */
import {
  createFakeTldwServer,
  HARNESS_WAIT,
  openSavedServerChat,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  SidebarServerChatButton,
  waitFor,
  waitForChatIdle,
  within,
  type FakeTldwServer
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"

const COMPLETIONS = "/api/v1/chat/completions"

const openSavedChat = async (server: FakeTldwServer) => {
  const chat = server.seedChat({
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
  return { chat, view }
}

describe("server refusals of server-settled turns through the chat transport", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("an Edit and send whose saved question changed elsewhere is refused, and the edited text is kept", async () => {
    const server = createFakeTldwServer({ nativeCompletion: true })
    const { chat, view } = await openSavedChat(server)
    const originals = structuredClone(chat.messages)
    const completion = server.holdRequest("POST", COMPLETIONS)

    await view.user.click(screen.getAllByRole("button", { name: "Edit" })[0])
    const editor = await waitFor(() => {
      const field = document.querySelector<HTMLTextAreaElement>("form[data-chat-message-editor] textarea")
      expect(field).not.toBeNull()
      return field as HTMLTextAreaElement
    }, HARNESS_WAIT)
    await view.user.clear(editor)
    await view.user.type(editor, "Edited question")
    await view.user.click(screen.getByRole("button", { name: "Save & Send" }))
    await completion.arrived
    // Another window edits the saved question before this one reaches the server.
    chat.messages[0].version += 1
    const changed = structuredClone(chat.messages)
    completion.release()
    await waitForChatIdle()

    // The server refused it with its full detail and saved nothing.
    expect(server.completionRequests()).toHaveLength(1)
    expect(chat.messages).toEqual(changed)
    expect(chat.messages.map((row) => row.id)).toEqual(originals.map((row) => row.id))
    // The editor stays open with the edited text and says why it was not sent.
    const editorAlert = await screen.findByRole("alert", {}, HARNESS_WAIT)
    expect(editorAlert).toHaveTextContent("The saved message has changed. Your edited text is kept.")
    expect(document.querySelector<HTMLTextAreaElement>("form[data-chat-message-editor] textarea")).toHaveValue("Edited question")
    // The turn is recorded as refused (not an uncertain outcome), with the edited text.
    const record = (await screen.findByText("Message not added", {}, HARNESS_WAIT)).closest("details") as HTMLElement
    expect(record).toHaveTextContent("Earlier messages to review (1)")
    expect(record).toHaveTextContent("Edited question")
    expect(screen.queryByText("Reply not confirmed")).toBeNull()
    expect(screen.queryByText(/could not confirm/i)).toBeNull()
  })

  it("a send whose latest message was continued elsewhere is held with Load latest messages and Branch here", async () => {
    const server = createFakeTldwServer({ nativeCompletion: true })
    const { chat, view } = await openSavedChat(server)
    const completion = server.holdRequest("POST", COMPLETIONS)

    await view.user.click(view.composer)
    await view.user.type(view.composer, "Next question")
    await view.user.keyboard("{Enter}")
    await completion.arrived
    // Another window continues the same reply first.
    const tip = chat.messages.at(-1)!.id
    chat.messages.push(
      { id: "elsewhere-q", role: "user", content: "Asked elsewhere", parent_message_id: tip, created_at: new Date().toISOString(), version: 1 },
      { id: "elsewhere-a", role: "assistant", content: "Answered elsewhere", parent_message_id: "elsewhere-q", created_at: new Date().toISOString(), version: 1 }
    )
    const rows = structuredClone(chat.messages)
    completion.release()
    await waitForChatIdle()

    expect(chat.messages).toEqual(rows)
    expect(server.completionRequests()[0].body).toMatchObject({ save_to_db: true, tldw_history_branch: false })
    const hold = (await screen.findByText("Updated in another window or tab", {}, HARNESS_WAIT)).closest("[role]") as HTMLElement
    expect(hold).toHaveTextContent("Your message was not added")
    expect(within(hold).getByRole("button", { name: "Load latest messages" })).toBeEnabled()
    expect(within(hold).getByRole("button", { name: "Branch here" })).toBeEnabled()
    // The question is not left in the transcript; the review keeps it.
    expect(screen.getAllByTestId("chat-message").map((row) => row.getAttribute("data-role"))).toEqual(["user", "assistant"])
    expect(screen.queryByText("Saved answer")).not.toBeNull()
    expect(within(screen.getByRole("region", { name: "Conversation history selection" })).getByText("Next question")).toBeInTheDocument()
  })
})
