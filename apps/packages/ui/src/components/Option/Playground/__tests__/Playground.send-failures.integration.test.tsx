// @vitest-environment jsdom
/**
 * Send failures and Stop in saved chats (#3107, UX G06: CC-02, CM-02).
 *
 * A send that fails, or that the user stops, must leave the question in the
 * transcript with a plain-language reason and the usual Retry and Switch model
 * actions, never the generic "Turn needs review" panel. Saved chats run both
 * ways: client-settled turns (the default harness server) and server-settled
 * turns (`nativeCompletion: true`, the real server's native_completion_v1).
 */
import {
  createFakeTldwServer,
  deferred,
  HARNESS_WAIT,
  visibleNotifications,
  within,
  openSavedServerChat,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  SidebarServerChatButton,
  simulatePageReload,
  waitFor,
  waitForChatIdle,
  type FakeTldwServer,
  type PlaygroundView
} from "./harness/playground-harness"
import { act } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

type Row = { role: string; text: string; interrupted?: boolean; stopped?: boolean }

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

const transcriptLines = () => transcript().map((row) => `${row.role}: ${row.text}`)

const serverRows = (server: FakeTldwServer, chatId: string) =>
  (server.chats.get(chatId)?.messages ?? []).map((row) => `${row.role}: ${row.content}`)

const openSeededChat = async (server: FakeTldwServer): Promise<PlaygroundView> => {
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
  return view
}

/** Type and send a message; resolves once the completion request was made. */
const sendAndAwaitRequest = async (view: PlaygroundView, text: string) => {
  const before = view.server.completionRequests().length
  await view.user.click(view.composer)
  await view.user.type(view.composer, text)
  await view.user.keyboard("{Enter}")
  await waitFor(
    () => expect(view.server.completionRequests().length).toBeGreaterThan(before),
    HARNESS_WAIT
  )
}

/** Type and send a message without waiting for the reply. */
const startSend = async (view: PlaygroundView, text: string) => {
  await view.user.click(view.composer)
  await view.user.type(view.composer, text)
  await view.user.keyboard("{Enter}")
  await waitFor(() => expect(view.composer).toHaveValue(""), HARNESS_WAIT)
}

const providerUnavailable = {
  status: 502,
  errorBody: {
    detail: {
      error_code: "provider_unavailable",
      message: "The chat service provider is currently unavailable."
    }
  }
}

/** The error reply shown for the failed send (not the composer's banner). */
const failureBubble = async () => {
  let bubble: HTMLElement | undefined
  await waitFor(() => {
    bubble = screen
      .getAllByRole("alert")
      .find(
        (alert) =>
          /Couldn't reach/.test(alert.textContent ?? "") &&
          within(alert).queryByRole("button", { name: "Retry same model" })
      )
    expect(bubble).toBeDefined()
  }, HARNESS_WAIT)
  return bubble as HTMLElement
}

describe("Playground send failures and Stop (#3107)", { timeout: 90_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it.each([
    { mode: "client-settled", nativeCompletion: false },
    { mode: "server-settled", nativeCompletion: true }
  ])(
    "CC-02 ($mode): a provider failure keeps the question with a plain-language error, Retry and Switch model",
    async ({ nativeCompletion }) => {
      const server = createFakeTldwServer({ nativeCompletion })
      server.planCompletion(providerUnavailable)
      server.planCompletion({ reply: "Complete answer" })
      const view = await openSeededChat(server)

      await sendAndAwaitRequest(view, "Follow-up question")
      await waitForChatIdle()

      const bubble = await failureBubble()
      expect(bubble).toHaveTextContent("Couldn't reach Custom OpenAI API (local-uat-chat)")
      expect(bubble).toHaveTextContent("The tldw server is fine")
      expect(within(bubble).getByRole("button", { name: "Switch model" })).toBeInTheDocument()
      // CC-N2: the composer banner does not repeat the error shown in the transcript.
      expect(screen.queryByTestId("playground-chat-error-banner")).toBeNull()
      await waitFor(
        () =>
          expect(transcriptLines().slice(0, 3)).toEqual([
            "user: Saved question",
            "assistant: Saved answer",
            "user: Follow-up question"
          ]),
        HARNESS_WAIT
      )
      expect(transcript()).toHaveLength(4)
      expect(transcript()[3]).toMatchObject({ role: "assistant", interrupted: true })
      expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
      // The question was saved once, before the provider failed.
      expect(serverRows(server, "saved-chat")).toEqual([
        "user: Saved question",
        "assistant: Saved answer",
        "user: Follow-up question"
      ])

      await view.user.click(within(await failureBubble()).getByRole("button", { name: "Retry same model" }))
      await waitFor(() => expect(server.completionRequests()).toHaveLength(2), HARNESS_WAIT)
      await waitForChatIdle()
      await waitFor(
        () =>
          expect(transcript()).toEqual([
            { role: "user", text: "Saved question" },
            { role: "assistant", text: "Saved answer" },
            { role: "user", text: "Follow-up question" },
            { role: "assistant", text: "Complete answer" }
          ]),
        HARNESS_WAIT
      )
      expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
      // Retry answers the saved question; it is not asked a second time.
      expect(serverRows(server, "saved-chat")).toEqual([
        "user: Saved question",
        "assistant: Saved answer",
        "user: Follow-up question",
        "assistant: Complete answer"
      ])
    }
  )

  it("CC-N2: a provider failure in a temporary chat names the provider and model, and is shown once", async () => {
    const server = createFakeTldwServer()
    server.planCompletion(providerUnavailable)
    const view = await renderPlayground({ server })
    act(() => useStoreMessageOption.getState().startNewChatDraft(true))

    await sendAndAwaitRequest(view, "Temporary question")
    await waitForChatIdle()

    const bubble = await failureBubble()
    expect(bubble).toHaveTextContent("Couldn't reach Custom OpenAI API (local-uat-chat)")
    expect(bubble).not.toHaveTextContent("Something went wrong while talking to your tldw server")
    expect(within(bubble).getByRole("button", { name: "Switch model" })).toBeInTheDocument()
    expect(screen.queryByTestId("playground-chat-error-banner")).toBeNull()
    // The model chip no longer calls a model whose provider is down "Healthy".
    expect(screen.getAllByText("Unreachable").length).toBeGreaterThan(0)
    expect(screen.queryByText("Healthy")).not.toBeInTheDocument()
  })

  it("CC-02 (server-settled): a request refused before the question is saved shows why, and Retry asks it again", async () => {
    const server = createFakeTldwServer({ nativeCompletion: true })
    server.planCompletion({
      ...providerUnavailable,
      rejectBeforeAdmission: true
    })
    server.planCompletion({ reply: "Complete answer" })
    const view = await openSeededChat(server)

    await sendAndAwaitRequest(view, "Follow-up question")
    await waitForChatIdle()

    const bubble = await failureBubble()
    expect(bubble).toHaveTextContent("Couldn't reach Custom OpenAI API (local-uat-chat)")
    expect(transcriptLines()).toEqual([
      "user: Saved question",
      "assistant: Saved answer",
      "user: Follow-up question",
      expect.stringContaining("assistant: ")
    ])
    expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
    expect(serverRows(server, "saved-chat")).toEqual(["user: Saved question", "assistant: Saved answer"])

    await view.user.click(within(bubble).getByRole("button", { name: "Retry same model" }))
    await waitFor(() => expect(server.completionRequests()).toHaveLength(2), HARNESS_WAIT)
    await waitForChatIdle()
    await waitFor(
      () =>
        expect(serverRows(server, "saved-chat")).toEqual([
          "user: Saved question",
          "assistant: Saved answer",
          "user: Follow-up question",
          "assistant: Complete answer"
        ]),
      HARNESS_WAIT
    )
    await waitFor(
      () =>
        expect(transcriptLines()).toEqual([
          "user: Saved question",
          "assistant: Saved answer",
          "user: Follow-up question",
          "assistant: Complete answer"
        ]),
      HARNESS_WAIT
    )
  })

  it("CM-02: Stop in a server-settled chat keeps the question and the partial answer, marked Stopped, with Retry", async () => {
    const server = createFakeTldwServer({ nativeCompletion: true })
    server.planCompletion({ chunks: ["Partial answer", " not wanted"], pauseAfterChunks: 1 })
    server.planCompletion({ reply: "Complete answer" })
    const view = await openSeededChat(server)

    await startSend(view, "Explain stopping")
    await waitFor(
      () => expect(transcript().some((row) => row.text.includes("Partial answer"))).toBe(true),
      HARNESS_WAIT
    )
    await view.user.click(screen.getByRole("button", { name: "Stop Streaming" }))
    await waitForChatIdle()

    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "Saved question" },
          { role: "assistant", text: "Saved answer" },
          { role: "user", text: "Explain stopping" },
          { role: "assistant", text: "Partial answer", interrupted: true, stopped: true }
        ]),
      HARNESS_WAIT
    )
    const marker = await screen.findByText("Stopped", { selector: "[data-interruption-label]" }, HARNESS_WAIT)
    const notice = marker.closest("[role='status']") as HTMLElement
    expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
    expect(visibleNotifications().join(" ")).not.toMatch(/Request cancelled|history admission|scope/)

    await view.user.click(within(notice).getByRole("button", { name: /retry/i }))
    await waitFor(() => expect(server.completionRequests()).toHaveLength(2), HARNESS_WAIT)
    await waitForChatIdle()
    await waitFor(
      () =>
        expect(transcript().slice(2)).toEqual([
          { role: "user", text: "Explain stopping" },
          { role: "assistant", text: "Complete answer" }
        ]),
      HARNESS_WAIT
    )
    // Retry answers the saved question; it is not asked a second time.
    expect(serverRows(server, "saved-chat").filter((row) => row === "user: Explain stopping")).toHaveLength(1)
    expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
  })

  it("CM-02: Stop before the reply starts keeps the saved question, marked Stopped before the reply, with Retry", async () => {
    const server = createFakeTldwServer()
    server.planCompletion({ chunks: ["Late answer"], pauseAfterChunks: 0 })
    server.planCompletion({ reply: "Complete answer" })
    const view = await openSeededChat(server)

    await startSend(view, "Explain stopping early")
    await waitFor(() => expect(server.completionRequests()).toHaveLength(1), HARNESS_WAIT)
    await view.user.click(await screen.findByRole("button", { name: "Stop Streaming" }, HARNESS_WAIT))
    await waitForChatIdle()

    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "Saved question" },
          { role: "assistant", text: "Saved answer" },
          { role: "user", text: "Explain stopping early" },
          { role: "assistant", text: "", interrupted: true, stopped: true }
        ]),
      HARNESS_WAIT
    )
    const marker = await screen.findByText("Stopped", { selector: "[data-interruption-label]" }, HARNESS_WAIT)
    const notice = marker.closest("[role='status']") as HTMLElement
    expect(notice).toHaveTextContent("You stopped this reply before it started.")
    expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
    expect(visibleNotifications().join(" ")).not.toMatch(/Error|scope/)

    await view.user.click(within(notice).getByRole("button", { name: /retry/i }))
    await waitFor(() => expect(server.completionRequests()).toHaveLength(2), HARNESS_WAIT)
    await waitForChatIdle()
    await waitFor(
      () =>
        expect(transcriptLines().slice(2)).toEqual([
          "user: Explain stopping early",
          "assistant: Complete answer"
        ]),
      HARNESS_WAIT
    )
    expect(serverRows(server, "saved-chat").filter((row) => row === "user: Explain stopping early")).toHaveLength(1)
  })

  it("CM-02: Stop before a server-settled reply starts puts the message back in the composer without an error", async () => {
    const server = createFakeTldwServer({ nativeCompletion: true })
    // The provider never starts answering, so the server never confirms the question.
    server.planCompletion({ chunks: ["Late answer"], pauseAfterChunks: 0 })
    const view = await openSeededChat(server)

    await startSend(view, "Never mind")
    await waitFor(() => expect(server.completionRequests()).toHaveLength(1), HARNESS_WAIT)
    await view.user.click(await screen.findByRole("button", { name: "Stop Streaming" }, HARNESS_WAIT))
    await waitForChatIdle()

    await waitFor(() => expect(view.composer).toHaveValue("Never mind"), HARNESS_WAIT)
    await waitFor(
      () => expect(transcriptLines()).toEqual(["user: Saved question", "assistant: Saved answer"]),
      HARNESS_WAIT
    )
    expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
    expect(visibleNotifications().join(" ")).not.toMatch(/Error|cancelled|scope/)
  })

  it("CM-02: Stop before the request is sent puts the message back in the composer without an error", async () => {
    const server = createFakeTldwServer()
    // Hold the history read that prepares the send, so Stop lands before dispatch.
    const prepared = deferred()
    let hold = false
    const serverFetch = server.fetch
    server.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url
      if (hold && /\/history\/selection$/.test(new URL(url, server.url).pathname)) await prepared.promise
      return serverFetch(input, init)
    }
    const view = await openSeededChat(server)

    hold = true
    await startSend(view, "Never mind")
    await view.user.click(await screen.findByRole("button", { name: "Stop Streaming" }, HARNESS_WAIT))
    hold = false
    prepared.resolve()
    await waitForChatIdle()

    await waitFor(() => expect(view.composer).toHaveValue("Never mind"), HARNESS_WAIT)
    expect(server.completionRequests()).toHaveLength(0)
    expect(visibleNotifications().join(" ")).not.toMatch(/request_config_scope_changed|stale_selection|Error/)
    expect(transcriptLines()).toEqual(["user: Saved question", "assistant: Saved answer"])
  })

  it.each([
    {
      outcome: "stopped",
      plan: { chunks: ["Partial answer", " not wanted"], pauseAfterChunks: 1 },
      reply: { role: "assistant", text: "Partial answer", interrupted: true, stopped: true } as Row
    },
    {
      outcome: "failed",
      plan: providerUnavailable,
      reply: { role: "assistant", text: expect.stringContaining("Couldn't reach"), interrupted: true } as Row
    }
  ])(
    "CM-02/CC-02 (server-settled): a $outcome reply is still shown once after a reload, never parked for review",
    async ({ outcome, plan, reply }) => {
      const server = createFakeTldwServer({ nativeCompletion: true })
      server.planCompletion(plan)
      const view = await openSeededChat(server)

      await sendAndAwaitRequest(view, "Question before reload")
      if (outcome === "stopped") {
        await waitFor(
          () => expect(transcript().some((row) => row.text.includes("Partial answer"))).toBe(true),
          HARNESS_WAIT
        )
        await view.user.click(screen.getByRole("button", { name: "Stop Streaming" }))
      }
      await waitForChatIdle()
      const expected = [
        { role: "user", text: "Saved question" },
        { role: "assistant", text: "Saved answer" },
        { role: "user", text: "Question before reload" },
        reply
      ]
      await waitFor(() => expect(transcript()).toEqual(expected), HARNESS_WAIT)
      // Let the retained record reach local storage before the page goes away.
      await new Promise((resolve) => setTimeout(resolve, 1_200))

      await simulatePageReload(view)
      await renderPlayground({
        server,
        keepBrowserState: true,
        extras: <SidebarServerChatButton chatId="saved-chat" title="Saved topic" />
      })

      await waitFor(() => expect(transcript()).toEqual(expected), HARNESS_WAIT)
      expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
      // The question was saved once.
      expect(
        serverRows(server, "saved-chat").filter((row) => row === "user: Question before reload")
      ).toHaveLength(1)
    }
  )

  it("harness: a server-settled saved-chat turn is admitted and settled by the server", async () => {
    const server = createFakeTldwServer({ nativeCompletion: true })
    const view = await openSeededChat(server)

    await startSend(view, "Follow-up question")
    await waitForChatIdle()

    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "Saved question" },
          { role: "assistant", text: "Saved answer" },
          { role: "user", text: "Follow-up question" },
          { role: "assistant", text: "Harness reply" }
        ]),
      HARNESS_WAIT
    )
    const request = server.completionRequests().at(-1)
    expect(request?.body?.save_to_db).toBe(true)
    expect(request?.body?.tldw_history_selection_v1).toBeTruthy()
    expect(serverRows(server, "saved-chat")).toEqual([
      "user: Saved question",
      "assistant: Saved answer",
      "user: Follow-up question",
      "assistant: Harness reply"
    ])
    // Only the server wrote the turn.
    expect(server.find("POST", "/api/v1/chats/saved-chat/messages")).toHaveLength(0)
    expect(screen.queryByText("Turn needs review")).not.toBeInTheDocument()
  })
})
