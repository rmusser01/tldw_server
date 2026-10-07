// @vitest-environment jsdom
/**
 * Interrupted replies on /chat (#3104, UX G03: CS-04, CS-N3).
 *
 * A reply can end early: the user stops it, the connection drops, the page
 * reloads, or the user leaves /chat while it streams. In every case the
 * question must stay in the transcript, the part of the reply that arrived must
 * be kept and marked ("Interrupted", or "Stopped" for a deliberate stop) with a
 * Retry, and no server chat may be created as a side effect. Since #3195 a fresh
 * chat on a connected server is owned by a server chat created by its first
 * send; that chat is the only one. These run against the real Playground
 * through the integration harness.
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
  waitFor,
  waitForChatIdle,
  within,
  type PlaygroundView
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { useConnectionStore } from "@/store/connection"
import { useStoreMessageOption } from "@/store/option"
import { getServerChatSaveStatus } from "@/store/server-chat-save-status"
import { resolveChatPersistenceKind } from "@/utils/chat-persistence-status"

type Row = { role: string; text: string; interrupted?: boolean; stopped?: boolean }

/** The visible transcript, with each reply's interruption marker. */
const transcript = (): Row[] =>
  useStoreMessageOption.getState().messages.map((message) => {
    const info = (message.generationInfo ?? {}) as Record<string, unknown>
    return {
      role: message.isBot ? "assistant" : "user",
      text: message.message,
      ...(info.interrupted ? { interrupted: true } : {}),
      ...(info.stopped ? { stopped: true } : {})
    }
  })

/** Type and send a message without waiting for the reply to finish. */
const startSend = async (view: PlaygroundView, text: string) => {
  await view.user.click(view.composer)
  await view.user.type(view.composer, text)
  await view.user.keyboard("{Enter}")
  await waitFor(() => expect(view.composer).toHaveValue(""), HARNESS_WAIT)
}

const waitForStreamedText = async (text: string) => {
  await waitFor(
    () => expect(transcript().some((row) => row.text.includes(text))).toBe(true),
    HARNESS_WAIT
  )
}

/** The interruption notice rendered on a kept partial reply. */
const interruptionNotice = async (label: "Interrupted" | "Stopped") => {
  const marker = await screen.findByText(label, { selector: "[data-interruption-label]" }, HARNESS_WAIT)
  return marker.closest("[role='status']") as HTMLElement
}

type StoredRow = { id: string; role: string; content: string; parent_message_id?: string | null }

/** A server chat's messages, with their parents. */
const serverRows = (server: ReturnType<typeof createFakeTldwServer>, chatId: string): StoredRow[] =>
  (server.chats.get(chatId)?.messages ?? []).map(({ id, role, content, parent_message_id }) => ({
    id,
    role,
    content,
    parent_message_id
  }))

/** The local chat database's messages, with their parents and interruption marker. */
const localRows = async () =>
  (await getHarnessDb().messages.toArray()).map((row) => ({
    id: String(row.id),
    role: String(row.role),
    content: String(row.content),
    parent_message_id: row.parent_message_id ?? null,
    interrupted: Boolean((row.generationInfo as Record<string, unknown> | undefined)?.interrupted)
  }))

/** Click Retry on the "Interrupted" notice and wait for the new reply to finish. */
const retryInterruptedReply = async (view: PlaygroundView, server: ReturnType<typeof createFakeTldwServer>) => {
  const before = server.completionRequests().length
  const notice = await interruptionNotice("Interrupted")
  await view.user.click(within(notice).getByRole("button", { name: /retry/i }))
  await waitFor(() => expect(server.completionRequests()).toHaveLength(before + 1), HARNESS_WAIT)
  await waitForChatIdle()
}

describe("Playground interrupted replies (#3104)", { timeout: 90_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("CS-04: a reload while a reply streams keeps the question and the partial answer, marked Interrupted", async () => {
    const server = createFakeTldwServer()
    // The second chunk never arrives: the page goes away mid-reply.
    server.planCompletion({ chunks: ["Partial answer", " that never finished"], pauseAfterChunks: 1 })
    const view = await renderPlayground({ server })

    await startSend(view, "Explain reloads")
    await waitForStreamedText("Partial answer")
    // Let the throttled checkpoint of the partial reply reach local storage.
    await new Promise((resolve) => setTimeout(resolve, 1_200))

    await simulatePageReload(view)
    await renderPlayground({ server, keepBrowserState: true })

    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "Explain reloads" },
          { role: "assistant", text: "Partial answer", interrupted: true }
        ]),
      HARNESS_WAIT
    )
    expect(await screen.findByText("Explain reloads", {}, HARNESS_WAIT)).toBeInTheDocument()
    const notice = await interruptionNotice("Interrupted")
    expect(within(notice).getByRole("button", { name: /retry/i })).toBeInTheDocument()
    expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
    // Not only on screen: a fresh chat is owned by a server chat from its first
    // send (#3195), which holds the question; the cut-off reply is kept from this
    // device, as for any server chat.
    const state = useStoreMessageOption.getState()
    expect(state.serverChatId).toBeTruthy()
    expect(
      server.chats.get(state.serverChatId as string)?.messages.map((message) => `${message.role}: ${message.content}`)
    ).toEqual(["user: Explain reloads"])
    expect(
      resolveChatPersistenceKind({
        temporaryChat: state.temporaryChat,
        serverChatId: state.serverChatId,
        serverSaveStatus: getServerChatSaveStatus(state.serverChatId)
      })
    ).toBe("serverFailed")
  })

  it("CS-04: a reply cut off by a dropped connection keeps the partial answer marked Interrupted, and Retry resends the question", async () => {
    const server = createFakeTldwServer()
    const resume = deferred()
    server.planCompletion({
      chunks: ["Partial answer", " lost"],
      pauseAfterChunks: 1,
      resume: resume.promise,
      end: "drop"
    })
    server.planCompletion({ reply: "Complete answer" })
    const view = await renderPlayground({ server })

    await startSend(view, "Explain drops")
    await waitForStreamedText("Partial answer")
    resume.resolve()
    await waitForChatIdle()

    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "Explain drops" },
          { role: "assistant", text: "Partial answer", interrupted: true }
        ]),
      HARNESS_WAIT
    )
    const notice = await interruptionNotice("Interrupted")
    expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()

    await view.user.click(within(notice).getByRole("button", { name: /retry/i }))
    await waitFor(() => expect(server.completionRequests()).toHaveLength(2), HARNESS_WAIT)
    await waitForChatIdle()

    // Retry asks the same question again, without the cut-off answer as context.
    expect(completionMessages(server.completionRequests()[1])).toEqual([
      { role: "user", content: "Explain drops" }
    ])
    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "Explain drops" },
          { role: "assistant", text: "Complete answer" }
        ]),
      HARNESS_WAIT
    )
  })

  it("CS-04: Stop keeps the partial answer and labels it Stopped", async () => {
    const server = createFakeTldwServer()
    server.planCompletion({ chunks: ["Partial answer", " not wanted"], pauseAfterChunks: 1 })
    const view = await renderPlayground({ server })

    await startSend(view, "Explain stopping")
    await waitForStreamedText("Partial answer")
    await view.user.click(screen.getByRole("button", { name: "Stop Streaming" }))
    await waitForChatIdle()

    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "Explain stopping" },
          { role: "assistant", text: "Partial answer", interrupted: true, stopped: true }
        ]),
      HARNESS_WAIT
    )
    const notice = await interruptionNotice("Stopped")
    expect(within(notice).getByRole("button", { name: /retry/i })).toBeInTheDocument()
    expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
  })

  it("CS-04: a message sent while Stop's kept turn is still loading waits in the composer, then sends", async () => {
    const server = createFakeTldwServer()
    server.planCompletion({ chunks: ["Never shown"], pauseAfterChunks: 0 })
    // Hold the history selection reads once Stop is pressed, so the view that
    // follows the stopped question stays "Loading selected history" (slow CI).
    const selectionRead = deferred()
    let holdSelectionReads = false
    const serverFetch = server.fetch
    server.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url
      if (holdSelectionReads && /\/history\/selection$/.test(new URL(url, server.url).pathname)) {
        await selectionRead.promise
      }
      return serverFetch(input, init)
    }
    const view = await renderPlayground({ server })

    await startSend(view, "Stop before any answer")
    await waitFor(() => expect(server.completionRequests()).toHaveLength(1), HARNESS_WAIT)
    holdSelectionReads = true
    await view.user.click(await screen.findByRole("button", { name: "Stop Streaming" }, HARNESS_WAIT))
    await waitForChatIdle()
    await screen.findByText("Loading selected history", {}, HARNESS_WAIT)

    // Sending now must not lose the message or fail with a raw error.
    await view.user.click(view.composer)
    await view.user.type(view.composer, "Ask again")
    await view.user.keyboard("{Enter}")
    await new Promise((resolve) => setTimeout(resolve, 200))
    expect(view.composer).toHaveValue("Ask again")
    expect(screen.queryByText("history_selection_not_ready")).not.toBeInTheDocument()
    expect(server.completionRequests()).toHaveLength(1)
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled()

    holdSelectionReads = false
    selectionRead.resolve()
    await waitFor(
      () => expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled(),
      HARNESS_WAIT
    )
    await view.user.click(screen.getByRole("button", { name: "Send message" }))
    await waitFor(() => expect(server.completionRequests()).toHaveLength(2), HARNESS_WAIT)
    await waitForChatIdle()
    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "Stop before any answer" },
          { role: "user", text: "Ask again" },
          { role: "assistant", text: "Harness reply" }
        ]),
      HARNESS_WAIT
    )
    expect(screen.queryByText("history_selection_not_ready")).not.toBeInTheDocument()
  })

  it("CS-04 / CS-N3: leaving /chat while a reply streams keeps the reply on return and creates no extra server chat", async () => {
    const server = createFakeTldwServer()
    const resume = deferred()
    server.planCompletion({
      chunks: ["Partial answer", " finished while away"],
      pauseAfterChunks: 1,
      resume: resume.promise
    })
    const view = await renderPlayground({ server })

    await startSend(view, "Explain navigation")
    await waitForStreamedText("Partial answer")
    // Leave /chat (in-app navigation keeps the app's stores), then the reply finishes.
    view.unmount()
    resume.resolve()
    await waitForChatIdle()

    await renderPlayground({ server, keepBrowserState: true })

    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "Explain navigation" },
          { role: "assistant", text: "Partial answer finished while away" }
        ]),
      HARNESS_WAIT
    )
    expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
    // Returning must not promote the chat into an empty or partial server chat.
    // Since #3195 the first send creates the chat's server owner; returning adds
    // no other chat, and the one chat holds the whole turn.
    await new Promise((resolve) => setTimeout(resolve, 1_000))
    expect(server.find("POST", /^\/api\/v1\/chats\/?$/)).toHaveLength(1)
    expect(
      [...server.chats.values()].map((chat) => chat.messages.map((message) => `${message.role}: ${message.content}`))
    ).toEqual([["user: Explain navigation", "assistant: Partial answer finished while away"]])
  })

  it("CS-04: an interrupted reply in a server chat keeps the question and is not labelled Saved on server", async () => {
    const server = createFakeTldwServer()
    server.seedChat({
      id: "saved-chat",
      title: "Saved topic",
      turns: [
        { role: "user", content: "Saved question" },
        { role: "assistant", content: "Saved answer" }
      ]
    })
    const resume = deferred()
    server.planCompletion({
      chunks: ["Partial answer", " lost"],
      pauseAfterChunks: 1,
      resume: resume.promise,
      end: "drop"
    })
    const view = await renderPlayground({
      server,
      extras: <SidebarServerChatButton chatId="saved-chat" title="Saved topic" />
    })
    await openSavedServerChat(view, { title: "Saved topic", lastMessage: "Saved answer" })

    await startSend(view, "Follow-up question")
    await waitForStreamedText("Partial answer")
    resume.resolve()
    await waitForChatIdle()

    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "Saved question" },
          { role: "assistant", text: "Saved answer" },
          { role: "user", text: "Follow-up question" },
          { role: "assistant", text: "Partial answer", interrupted: true }
        ]),
      HARNESS_WAIT
    )
    // The question reached the server; the cut-off reply did not.
    expect(
      server.chats.get("saved-chat")?.messages.map((message) => `${message.role}: ${message.content}`)
    ).toEqual(["user: Saved question", "assistant: Saved answer", "user: Follow-up question"])
    const state = useStoreMessageOption.getState()
    expect(
      resolveChatPersistenceKind({
        temporaryChat: state.temporaryChat,
        serverChatId: state.serverChatId,
        serverSaveStatus: getServerChatSaveStatus(state.serverChatId)
      })
    ).not.toBe("server")
  })

  it("CS-04: a reload while a server-chat reply streams keeps the question and the partial answer from this device", async () => {
    const server = createFakeTldwServer()
    server.seedChat({
      id: "saved-chat",
      title: "Saved topic",
      turns: [
        { role: "user", content: "Saved question" },
        { role: "assistant", content: "Saved answer" }
      ]
    })
    server.planCompletion({ chunks: ["Partial answer", " lost"], pauseAfterChunks: 1 })
    const view = await renderPlayground({
      server,
      extras: <SidebarServerChatButton chatId="saved-chat" title="Saved topic" />
    })
    await openSavedServerChat(view, { title: "Saved topic", lastMessage: "Saved answer" })

    await startSend(view, "Follow-up question")
    await waitForStreamedText("Partial answer")
    await new Promise((resolve) => setTimeout(resolve, 1_200))

    await simulatePageReload(view)
    await renderPlayground({ server, keepBrowserState: true })

    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "Saved question" },
          { role: "assistant", text: "Saved answer" },
          { role: "user", text: "Follow-up question" },
          { role: "assistant", text: "Partial answer", interrupted: true }
        ]),
      HARNESS_WAIT
    )
    await interruptionNotice("Interrupted")
    expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
    const state = useStoreMessageOption.getState()
    expect(state.serverChatId).toBe("saved-chat")
    expect(
      resolveChatPersistenceKind({
        temporaryChat: state.temporaryChat,
        serverChatId: state.serverChatId,
        serverSaveStatus: getServerChatSaveStatus(state.serverChatId)
      })
    ).toBe("serverFailed")
  })

  // Retry answers the question that is already saved; it never saves it again.
  describe("Retry after a dropped reply answers the saved question once", () => {
    const planDroppedThenComplete = (server: ReturnType<typeof createFakeTldwServer>) => {
      const resume = deferred()
      server.planCompletion({
        chunks: ["Partial answer", " lost"],
        pauseAfterChunks: 1,
        resume: resume.promise,
        end: "drop"
      })
      server.planCompletion({ reply: "Complete answer" })
      return resume
    }

    it("in a saved server chat: one copy of the question on screen and on the server, answered by the new reply", async () => {
      const server = createFakeTldwServer()
      server.seedChat({
        id: "saved-chat",
        title: "Saved topic",
        turns: [
          { role: "user", content: "Saved question" },
          { role: "assistant", content: "Saved answer" }
        ]
      })
      const resume = planDroppedThenComplete(server)
      const view = await renderPlayground({
        server,
        extras: <SidebarServerChatButton chatId="saved-chat" title="Saved topic" />
      })
      await openSavedServerChat(view, { title: "Saved topic", lastMessage: "Saved answer" })

      await startSend(view, "Follow-up question")
      await waitForStreamedText("Partial answer")
      resume.resolve()
      await waitForChatIdle()
      await retryInterruptedReply(view, server)

      const expectedTranscript = [
        { role: "user", text: "Saved question" },
        { role: "assistant", text: "Saved answer" },
        { role: "user", text: "Follow-up question" },
        { role: "assistant", text: "Complete answer" }
      ]
      await waitFor(() => expect(transcript()).toEqual(expectedTranscript), HARNESS_WAIT)
      const rows = serverRows(server, "saved-chat")
      expect(rows.map((row) => `${row.role}: ${row.content}`)).toEqual([
        "user: Saved question",
        "assistant: Saved answer",
        "user: Follow-up question",
        "assistant: Complete answer"
      ])
      // The new reply answers the question saved by the first send.
      expect(rows[3].parent_message_id).toBe(rows[2].id)
      // Retry asked the same question, with the same context, without the cut-off answer.
      expect(completionMessages(server.completionRequests()[1])).toEqual(
        completionMessages(server.completionRequests()[0])
      )
      // Nothing waits for review or keeps the cut-off reply on screen.
      await new Promise((resolve) => setTimeout(resolve, 500))
      expect(transcript()).toEqual(expectedTranscript)
      expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
      const state = useStoreMessageOption.getState()
      expect(
        resolveChatPersistenceKind({
          temporaryChat: state.temporaryChat,
          serverChatId: state.serverChatId,
          serverSaveStatus: getServerChatSaveStatus(state.serverChatId)
        })
      ).toBe("server")
    })

    it("in a fresh chat, owned by the server chat its first send created", async () => {
      const server = createFakeTldwServer()
      const resume = planDroppedThenComplete(server)
      const view = await renderPlayground({ server })

      await startSend(view, "Explain drops")
      await waitForStreamedText("Partial answer")
      resume.resolve()
      await waitForChatIdle()
      await retryInterruptedReply(view, server)

      await waitFor(
        () =>
          expect(transcript()).toEqual([
            { role: "user", text: "Explain drops" },
            { role: "assistant", text: "Complete answer" }
          ]),
        HARNESS_WAIT
      )
      expect(server.find("POST", /^\/api\/v1\/chats\/?$/)).toHaveLength(1)
      const chatId = useStoreMessageOption.getState().serverChatId as string
      const rows = serverRows(server, chatId)
      expect(rows.map((row) => `${row.role}: ${row.content}`)).toEqual([
        "user: Explain drops",
        "assistant: Complete answer"
      ])
      expect(rows[1].parent_message_id).toBe(rows[0].id)
    })

    it("in a local chat: one copy of the question, and the cut-off reply stays as an alternative", async () => {
      const server = createFakeTldwServer()
      const resume = planDroppedThenComplete(server)
      const view = await renderPlayground({ server })
      // A fresh chat stays on this device when it cannot get a server owner
      // before its first send (here: the offline bypass).
      useConnectionStore.setState((store) => ({ state: { ...store.state, offlineBypass: true } }))

      await startSend(view, "Explain drops")
      await waitForStreamedText("Partial answer")
      resume.resolve()
      await waitForChatIdle()
      await waitFor(
        () =>
          expect(transcript()).toEqual([
            { role: "user", text: "Explain drops" },
            { role: "assistant", text: "Partial answer", interrupted: true }
          ]),
        HARNESS_WAIT
      )
      expect(useStoreMessageOption.getState().serverChatId).toBeFalsy()
      await retryInterruptedReply(view, server)

      await waitFor(
        () =>
          expect(transcript()).toEqual([
            { role: "user", text: "Explain drops" },
            { role: "assistant", text: "Complete answer" }
          ]),
        HARNESS_WAIT
      )
      expect(completionMessages(server.completionRequests()[1])).toEqual([
        { role: "user", content: "Explain drops" }
      ])
      const rows = await localRows()
      const questions = rows.filter((row) => row.role === "user")
      expect(questions.map((row) => row.content)).toEqual(["Explain drops"])
      // Both replies answer that one question: the new one, and the cut-off
      // one kept as an alternative.
      const replies = rows.filter((row) => row.role === "assistant")
      expect(replies.map(({ content, interrupted, parent_message_id }) => ({ content, interrupted, parent_message_id }))).toEqual(
        expect.arrayContaining([
          { content: "Partial answer", interrupted: true, parent_message_id: questions[0].id },
          { content: "Complete answer", interrupted: false, parent_message_id: questions[0].id }
        ])
      )
      expect(replies).toHaveLength(2)
      const reply = useStoreMessageOption.getState().messages.at(-1)
      expect(reply?.variants?.map((variant) => variant.message)).toEqual(
        expect.arrayContaining(["Partial answer", "Complete answer"])
      )
      // The chat never moved to the server.
      expect(server.find("POST", /^\/api\/v1\/chats\/?$/)).toHaveLength(0)
      expect(server.chats.size).toBe(0)
    })
  })
})
