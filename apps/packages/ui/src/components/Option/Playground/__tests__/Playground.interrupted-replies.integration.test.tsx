// @vitest-environment jsdom
/**
 * Interrupted replies on /chat (#3104, UX G03: CS-04, CS-N3).
 *
 * A reply can end early: the user stops it, the connection drops, the page
 * reloads, or the user leaves /chat while it streams. In every case the
 * question must stay in the transcript, the part of the reply that arrived must
 * be kept and marked ("Interrupted", or "Stopped" for a deliberate stop) with a
 * Retry, and no server chat may be created as a side effect. These run against
 * the real Playground through the integration harness.
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

const localRows = async () =>
  (await getHarnessDb().messages.toArray()).map((row) => ({
    role: row.role,
    content: row.content,
    interrupted: Boolean((row.generationInfo as Record<string, unknown> | undefined)?.interrupted)
  }))

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
    // The question and the partial answer are in the local chat, not only on screen.
    expect(await localRows()).toEqual(
      expect.arrayContaining([
        { role: "user", content: "Explain reloads", interrupted: false },
        { role: "assistant", content: "Partial answer", interrupted: true }
      ])
    )
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

  it("CS-04 / CS-N3: leaving /chat while a reply streams keeps the reply on return and creates no server chat", async () => {
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
    await new Promise((resolve) => setTimeout(resolve, 1_000))
    expect(server.find("POST", /^\/api\/v1\/chats\/?$/)).toEqual([])
    expect([...server.chats.values()]).toEqual([])
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
})
