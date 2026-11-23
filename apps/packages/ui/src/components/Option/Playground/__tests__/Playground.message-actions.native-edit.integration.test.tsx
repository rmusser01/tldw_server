// @vitest-environment jsdom
/**
 * D7 P5 (#3101, Wave 8 lane 8A): Edit -> "Save & Send" in a saved server chat
 * whose server settles turns (`native_completion_v1`) is settled by the server.
 *
 * The edited question goes out once, on /chat/completions with `save_to_db`
 * and `tldw_history_edit_v1`; the client never appends it through
 * `POST /chats/{id}/messages`. The server saves the reply with its generation
 * metadata, so a reply cut off at the output limit keeps its marker after the
 * view follows it from the server.
 */
import {
  completionMessages,
  createFakeTldwServer,
  HARNESS_WAIT,
  openSavedServerChat,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  SidebarServerChatButton,
  simulatePageReload,
  visibleNotifications,
  waitFor,
  waitForChatIdle,
  within
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"

const lastReply = () =>
  screen.getAllByTestId("chat-message").filter((row) => row.getAttribute("data-role") === "assistant").at(-1)!

describe("server-settled Edit and send (D7 P5)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("P5: the server settles an edited question, and a reply at the output limit keeps its marker", async () => {
    const server = createFakeTldwServer({ nativeCompletion: true })
    const chat = server.seedChat({
      id: "saved-chat",
      title: "Saved topic",
      turns: [
        { role: "user", content: "Saved question" },
        { role: "assistant", content: "Saved answer" }
      ]
    })
    const originals = structuredClone(chat.messages)
    const view = await renderPlayground({
      server,
      extras: <SidebarServerChatButton chatId="saved-chat" title="Saved topic" />
    })
    await openSavedServerChat(view, { title: "Saved topic", lastMessage: "Saved answer" })
    server.planCompletion({ chunks: ["Edited answer that was cut off mid-"], finishReason: "length" })

    // The first Edit action belongs to the first (user) message.
    await view.user.click(screen.getAllByRole("button", { name: "Edit" })[0])
    const editor = await waitFor(() => {
      const field = document.querySelector<HTMLTextAreaElement>("form[data-chat-message-editor] textarea")
      expect(field).not.toBeNull()
      return field as HTMLTextAreaElement
    }, HARNESS_WAIT)
    await view.user.clear(editor)
    await view.user.type(editor, "Edited question")
    await view.user.click(screen.getByRole("button", { name: "Save & Send" }))
    await waitFor(() => expect(server.completionRequests()).toHaveLength(1), HARNESS_WAIT)
    await waitForChatIdle()

    // One completion carries the edit; the server saves it.
    const request = server.completionRequests()[0].body!
    expect(request).toMatchObject({
      save_to_db: true,
      conversation_id: "saved-chat",
      tldw_history_branch: true,
      tldw_history_edit_v1: { version: 1, message_id: originals[0].id, message_revision: String(originals[0].version) },
      tldw_history_selection_v1: { cursor: { kind: "before_message", message_id: originals[0].id } }
    })
    expect(completionMessages({ body: request }).at(-1)).toEqual({ role: "user", content: "Edited question" })
    // The client does not save the question or the reply itself.
    expect(server.find("POST", /\/chats\/saved-chat\/messages$/)).toEqual([])

    // The originals stay; the edited question branches from the same parent,
    // and the server saved the reply with its output-limit status.
    expect(chat.messages.filter((row) => originals.some((original) => original.id === row.id))).toEqual(originals)
    const added = chat.messages.filter((row) => !originals.some((original) => original.id === row.id))
    expect(added).toEqual([
      expect.objectContaining({ role: "user", content: "Edited question", parent_message_id: originals[0].parent_message_id }),
      expect.objectContaining({ role: "assistant", content: "Edited answer that was cut off mid-", parent_message_id: added[0]?.id })
    ])
    expect(added[1].metadata_extra).toMatchObject({ generation_status: "length" })

    // The view follows the saved reply, which keeps its output-limit marker.
    await waitFor(() => {
      expect(
        server
          .find("POST", /\/history\/selection$/)
          .some((capture) => capture.body?.view?.cursor?.message_id === added[1].id)
      ).toBe(true)
      expect(lastReply()).toHaveTextContent("Edited answer that was cut off mid-")
    }, HARNESS_WAIT)
    const notice = await within(lastReply()).findByRole("status", { name: "Output limit reached" }, HARNESS_WAIT)
    expect(within(notice).getByRole("button", { name: "Continue from partial" })).toBeEnabled()
    expect(screen.getByText("Edited question")).toBeInTheDocument()
    expect(screen.queryByText("Saved question")).not.toBeInTheDocument()
    expect(visibleNotifications()).toEqual([])

    // The marker is the server's: it is still there after a reload.
    await simulatePageReload(view)
    const reloaded = await renderPlayground({
      server,
      keepBrowserState: true,
      extras: <SidebarServerChatButton chatId="saved-chat" title="Saved topic" />
    })
    await openSavedServerChat(reloaded, { title: "Saved topic", lastMessage: "Edited answer that was cut off mid-" })
    await within(lastReply()).findByRole("status", { name: "Output limit reached" }, HARNESS_WAIT)
    expect(server.completionRequests()).toHaveLength(1)
  })
})
