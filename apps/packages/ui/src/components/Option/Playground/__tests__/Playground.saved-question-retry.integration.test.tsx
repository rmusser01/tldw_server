// @vitest-environment jsdom
/**
 * Retry of a saved question in a plain server chat (Wave 8 lane 8B, D7 P5, #3101).
 *
 * A turn can end with the user's question saved on the server and no reply: a
 * Stop before the first token, or a provider error after the question was
 * admitted. The user reviews the saved question and retries it. The retry is
 * settled by the server through `tldw_history_server_existing_input_v1`, so
 * the question is never saved twice and the reply keeps its model and
 * output-limit marker. The active System/Prompt instructions come along as
 * system-only messages (decision 22); a server that can't take them refuses
 * before anything is sent and keeps the question.
 */
import {
  createFakeTldwServer,
  deferred,
  FAKE_MODEL,
  getHarnessDb,
  HARNESS_WAIT,
  openSavedServerChat,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  SidebarServerChatButton,
  waitFor,
  waitForChatIdle,
  within,
  type FakeTldwServer,
  type PlaygroundView
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { setSystemPromptForNonRagOption } from "@/services/tldw-server"
import { useStoreMessageOption } from "@/store/option"

const transcript = () =>
  useStoreMessageOption.getState().messages.map((message) => ({
    role: message.isBot ? "assistant" : "user",
    text: message.message
  }))

const serverRows = (server: FakeTldwServer, chatId: string) =>
  (server.chats.get(chatId)?.messages ?? []).map(({ id, role, content, parent_message_id }) => ({
    id,
    role,
    content,
    parent_message_id
  }))

const lastReply = () =>
  screen.getAllByTestId("chat-message").filter((row) => row.getAttribute("data-role") === "assistant").at(-1)!

/**
 * Send `text`, Stop before any reply text arrives, and wait until the turn
 * ends. The held first reply is released by the returned callback.
 */
const stopBeforeFirstToken = async (view: PlaygroundView, text: string) => {
  const held = deferred()
  view.server.planCompletion({ chunks: ["never sent"], pauseAfterChunks: 0, resume: held.promise })
  await view.user.click(view.composer)
  await view.user.type(view.composer, text)
  await view.user.keyboard("{Enter}")
  await waitFor(() => expect(view.server.completionRequests()).toHaveLength(1), HARNESS_WAIT)
  await view.user.click(screen.getByRole("button", { name: "Stop Streaming" }))
  await waitForChatIdle()
  return held.resolve
}

/** Open the saved-question review and press Retry. */
const retrySavedQuestion = async (view: PlaygroundView) => {
  await view.user.click(await screen.findByRole("button", { name: "Review saved question" }, HARNESS_WAIT))
  const review = await screen.findByRole("group", { name: "Saved question review" }, HARNESS_WAIT)
  await view.user.click(within(review).getByRole("button", { name: "Retry saved question" }))
  return review
}

const openChat = async (server: FakeTldwServer, saved: boolean) => {
  if (saved)
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
    ...(saved ? { extras: <SidebarServerChatButton chatId="saved-chat" title="Saved topic" /> } : {})
  })
  if (saved) await openSavedServerChat(view, { title: "Saved topic", lastMessage: "Saved answer" })
  return view
}

describe("Playground saved-question Retry in a plain server chat (8B)", { timeout: 90_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it.each([
    { label: "a fresh chat", saved: false },
    { label: "a saved server chat", saved: true }
  ])("CM-02 with server-settled turns: Stop before any text in $label, then Retry answers the saved question once", async ({ saved }) => {
    const server = createFakeTldwServer({ nativeCompletion: true })
    const view = await openChat(server, saved)
    const earlier = saved
      ? [
          { role: "user", text: "Saved question" },
          { role: "assistant", text: "Saved answer" }
        ]
      : []
    const release = await stopBeforeFirstToken(view, "Stop me early")
    try {
      const chatId = String(useStoreMessageOption.getState().serverChatId)
      // The server holds the question once and no reply.
      expect(serverRows(server, chatId).map((row) => `${row.role}: ${row.content}`)).toEqual([
        ...earlier.map((row) => `${row.role}: ${row.text}`),
        "user: Stop me early"
      ])
      await waitFor(() => expect(transcript()).toEqual([...earlier, { role: "user", text: "Stop me early" }]), HARNESS_WAIT)
      expect(view.composer).toHaveValue("")

      server.planCompletion({ reply: "Answer after retry" })
      await retrySavedQuestion(view)
      await waitFor(() => expect(server.completionRequests()).toHaveLength(2), HARNESS_WAIT)
      await waitForChatIdle()

      const retry = server.completionRequests()[1].body!
      expect(retry.tldw_history_server_existing_input_v1).toMatchObject({ cursor: { kind: "after_message" } })
      expect(retry.tldw_history_selection_v1).toBeUndefined()
      expect(retry.messages.filter((row: { role: string }) => row.role !== "system")).toEqual([])
      const rows = serverRows(server, chatId)
      const question = rows.filter((row) => row.content === "Stop me early")
      expect(question).toHaveLength(1)
      expect(rows.at(-1)).toMatchObject({ role: "assistant", content: "Answer after retry", parent_message_id: question[0].id })
      await waitFor(
        () =>
          expect(transcript()).toEqual([
            ...earlier,
            { role: "user", text: "Stop me early" },
            { role: "assistant", text: "Answer after retry" }
          ]),
        HARNESS_WAIT
      )
    } finally {
      release()
    }
  })

  it("a failure after the question was saved, with no reply text, is retried the same way", async () => {
    const server = createFakeTldwServer({ nativeCompletion: true })
    const view = await openChat(server, false)
    // The server admits the question, then the stream fails before any text.
    const fail = deferred()
    server.planCompletion({ chunks: ["never sent"], pauseAfterChunks: 0, resume: fail.promise, end: "drop" })
    await view.user.click(view.composer)
    await view.user.type(view.composer, "Fail after saving")
    await view.user.keyboard("{Enter}")
    await waitFor(() => expect(server.completionRequests()).toHaveLength(1), HARNESS_WAIT)
    // Fail only once this device has recorded the admission (the question is saved).
    await waitFor(async () => {
      const rows = await getHarnessDb().historySelections.toArray()
      expect(rows.some((row) => Object.values(row.pending_turns ?? {}).some((turn) => Boolean(turn.admission)))).toBe(true)
    }, HARNESS_WAIT)
    fail.resolve()
    await waitForChatIdle()
    await waitFor(() => expect(transcript()).toEqual([{ role: "user", text: "Fail after saving" }]), HARNESS_WAIT)

    server.planCompletion({ reply: "Answer after the failure" })
    await retrySavedQuestion(view)
    await waitFor(() => expect(server.completionRequests()).toHaveLength(2), HARNESS_WAIT)
    await waitForChatIdle()
    expect(server.completionRequests()[1].body!.tldw_history_server_existing_input_v1).toBeDefined()
    const chatId = String(useStoreMessageOption.getState().serverChatId)
    expect(serverRows(server, chatId).map((row) => `${row.role}: ${row.content}`)).toEqual([
      "user: Fail after saving",
      "assistant: Answer after the failure"
    ])
  })

  it("a Retry reply at the output limit keeps its marker and model after the view follows it", async () => {
    const server = createFakeTldwServer({ nativeCompletion: true })
    const view = await openChat(server, false)
    const release = await stopBeforeFirstToken(view, "Explain at length")
    try {
      server.planCompletion({ chunks: ["A long answer that was cut"], finishReason: "length" })
      await retrySavedQuestion(view)
      await waitFor(() => expect(server.completionRequests()).toHaveLength(2), HARNESS_WAIT)
      await waitForChatIdle()

      const chatId = String(useStoreMessageOption.getState().serverChatId)
      expect(serverRows(server, chatId).at(-1)).toMatchObject({ role: "assistant", content: "A long answer that was cut" })
      expect(server.chats.get(chatId)?.messages.at(-1)?.metadata_extra).toMatchObject({
        generation_status: "length",
        model_id: FAKE_MODEL
      })
      await waitFor(() => expect(lastReply()).toHaveTextContent("A long answer that was cut"), HARNESS_WAIT)
      expect(await within(lastReply()).findByRole("status", { name: "Output limit reached" }, {}, HARNESS_WAIT)).toBeVisible()
      expect(within(lastReply()).getByTestId("message-model-badge")).toHaveTextContent(FAKE_MODEL)
    } finally {
      release()
    }
  })

  it("carries the default System instruction when the server advertises it", async () => {
    const server = createFakeTldwServer({ nativeCompletion: true })
    const view = await openChat(server, false)
    // Set after the harness seeds browser storage; the first send already uses it.
    await setSystemPromptForNonRagOption("Answer like a pirate.")
    const release = await stopBeforeFirstToken(view, "Where is the treasure?")
    try {
      server.planCompletion({ reply: "Arr, buried." })
      await retrySavedQuestion(view)
      await waitFor(() => expect(server.completionRequests()).toHaveLength(2), HARNESS_WAIT)
      await waitForChatIdle()

      expect(server.completionRequests()[1].body!.messages).toEqual([{ role: "system", content: "Answer like a pirate." }])
      const chatId = String(useStoreMessageOption.getState().serverChatId)
      expect(serverRows(server, chatId).map((row) => row.role)).toEqual(["user", "assistant"])
    } finally {
      release()
    }
  })

  it("an older server refuses the Retry before sending while an instruction is active, and keeps the question", async () => {
    const server = createFakeTldwServer({ nativeCompletion: true, existingInputInstructions: false })
    const view = await openChat(server, false)
    await setSystemPromptForNonRagOption("Answer like a pirate.")
    const release = await stopBeforeFirstToken(view, "Where is the treasure?")
    try {
      await view.user.click(await screen.findByRole("button", { name: "Review saved question" }, HARNESS_WAIT))
      const review = await screen.findByRole("group", { name: "Saved question review" }, HARNESS_WAIT)
      expect(within(review).getByText("Answer like a pirate.")).toBeVisible()
      expect(within(review).getByRole("alert")).toHaveTextContent(
        "This server can't keep your System/Prompt instructions on a retry"
      )
      expect(within(review).getByRole("button", { name: "Retry saved question" })).toBeDisabled()
      expect(server.completionRequests()).toHaveLength(1)
      const chatId = String(useStoreMessageOption.getState().serverChatId)
      expect(serverRows(server, chatId).map((row) => `${row.role}: ${row.content}`)).toEqual(["user: Where is the treasure?"])
      expect(transcript()).toEqual([{ role: "user", text: "Where is the treasure?" }])
    } finally {
      release()
    }
  })
})
