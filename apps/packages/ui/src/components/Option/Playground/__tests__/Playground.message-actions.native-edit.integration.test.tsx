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
  deferred,
  getHarnessDb,
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
  within,
  type FakeChat,
  type FakeChatMessage,
  type FakeTldwServer,
  type PlaygroundView
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

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

/**
 * 8A x 8B (#3101): an edited question the server admitted but never answered.
 * A Stop before any reply text (whitespace counts as none, as in CM-02), or a
 * provider failure after admission, leaves the edited question saved on the
 * server with no reply. The view must follow that admitted question (not stay
 * on the edit view, which still shows the original), and "Review saved
 * question" retries it through existing input, so it is saved once and the
 * reply hangs under it.
 */
describe("an edited question stopped or failed before its reply (8A x 8B)", { timeout: 90_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  const transcript = () =>
    useStoreMessageOption.getState().messages.map((message) => ({
      role: message.isBot ? "assistant" : "user",
      text: message.message
    }))

  const openEditedChat = async () => {
    const server = createFakeTldwServer({ nativeCompletion: true })
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
    return { server, chat, originals: structuredClone(chat.messages), view }
  }

  /** Edit the first question and press Save & Send. */
  const editAndSend = async (view: PlaygroundView, text: string) => {
    await view.user.click(screen.getAllByRole("button", { name: "Edit" })[0])
    const editor = await waitFor(() => {
      const field = document.querySelector<HTMLTextAreaElement>("form[data-chat-message-editor] textarea")
      expect(field).not.toBeNull()
      return field as HTMLTextAreaElement
    }, HARNESS_WAIT)
    await view.user.clear(editor)
    await view.user.type(editor, text)
    await view.user.click(screen.getByRole("button", { name: "Save & Send" }))
  }

  /** Retry the saved question and check it was answered once, under the edited question. */
  const retryAnswersEditedQuestionOnce = async (
    view: PlaygroundView,
    server: FakeTldwServer,
    chat: FakeChat,
    originals: FakeChatMessage[]
  ) => {
    server.planCompletion({ reply: "Answer to the edit" })
    await view.user.click(await screen.findByRole("button", { name: "Review saved question" }, HARNESS_WAIT))
    const review = await screen.findByRole("group", { name: "Saved question review" }, HARNESS_WAIT)
    expect(review).toHaveTextContent("Edited question")
    await view.user.click(within(review).getByRole("button", { name: "Retry saved question" }))
    await waitFor(() => expect(server.completionRequests()).toHaveLength(2), HARNESS_WAIT)
    await waitForChatIdle()

    const retry = server.completionRequests()[1].body!
    expect(retry.tldw_history_server_existing_input_v1).toMatchObject({ cursor: { kind: "after_message" } })
    expect(retry.tldw_history_edit_v1).toBeUndefined()
    expect(retry.messages.filter((row: { role: string }) => row.role !== "system")).toEqual([])
    const added = chat.messages.filter((row) => !originals.some((original) => original.id === row.id))
    expect(added).toEqual([
      expect.objectContaining({ role: "user", content: "Edited question", parent_message_id: originals[0].parent_message_id }),
      expect.objectContaining({ role: "assistant", content: "Answer to the edit", parent_message_id: added[0]?.id })
    ])
    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "Edited question" },
          { role: "assistant", text: "Answer to the edit" }
        ]),
      HARNESS_WAIT
    )
    expect(visibleNotifications()).toEqual([])
  }

  it.each([
    { label: "no output", initialText: "" },
    { label: "whitespace-only output", initialText: " \n\t" },
    { label: "blank-line output", initialText: "\n\n" }
  ])("Stop before any text ($label) keeps the edited question in view, and Retry answers it once", async ({ initialText }) => {
    const { server, chat, originals, view } = await openEditedChat()
    const held = deferred()
    server.planCompletion({
      chunks: [...(initialText ? [initialText] : []), "never sent"],
      pauseAfterChunks: initialText ? 1 : 0,
      resume: held.promise
    })
    try {
      await editAndSend(view, "Edited question")
      await waitFor(() => expect(server.completionRequests()).toHaveLength(1), HARNESS_WAIT)
      expect(server.completionRequests()[0].body!.tldw_history_edit_v1).toBeDefined()
      // Whitespace counts as no reply text; wait until the turn has read it.
      if (initialText) await waitFor(() => expect(useStoreMessageOption.getState().isProcessing).toBe(true), HARNESS_WAIT)
      await view.user.click(screen.getByRole("button", { name: "Stop Streaming" }))
      await waitForChatIdle()

      // The server holds the edited question once, beside the originals, and no reply.
      expect(chat.messages.filter((row) => !originals.some((original) => original.id === row.id))).toEqual([
        expect.objectContaining({ role: "user", content: "Edited question", parent_message_id: originals[0].parent_message_id })
      ])
      // The view follows the admitted edited question, not the edit view of the original.
      await waitFor(() => expect(transcript()).toEqual([{ role: "user", text: "Edited question" }]), HARNESS_WAIT)
      expect(screen.queryByText("Saved question")).not.toBeInTheDocument()

      await retryAnswersEditedQuestionOnce(view, server, chat, originals)
    } finally {
      held.resolve()
    }
  })

  it("a provider failure after the edited question was admitted is retried the same way", async () => {
    const { server, chat, originals, view } = await openEditedChat()
    const fail = deferred()
    server.planCompletion({ chunks: ["never sent"], pauseAfterChunks: 0, resume: fail.promise, end: "drop" })
    await editAndSend(view, "Edited question")
    await waitFor(() => expect(server.completionRequests()).toHaveLength(1), HARNESS_WAIT)
    // Fail only once this device has recorded the admission (the edited question is saved).
    await waitFor(async () => {
      const rows = await getHarnessDb().historySelections.toArray()
      expect(rows.some((row) => Object.values(row.pending_turns ?? {}).some((turn) => Boolean(turn.admission)))).toBe(true)
    }, HARNESS_WAIT)
    fail.resolve()
    await waitForChatIdle()
    await waitFor(() => expect(transcript()).toEqual([{ role: "user", text: "Edited question" }]), HARNESS_WAIT)

    await retryAnswersEditedQuestionOnce(view, server, chat, originals)
  })
})
