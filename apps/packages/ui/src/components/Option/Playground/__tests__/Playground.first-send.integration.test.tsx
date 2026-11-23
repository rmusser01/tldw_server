// @vitest-environment jsdom
/**
 * CM-14 (#3101): the first send in a new chat. Since D7 P4 a fresh chat on a
 * connected server is created by its first send, so the question waited for
 * the server create and the new chat's first load before it appeared, and the
 * load flashed "Loading selected history" over an empty chat. These run
 * against the real Playground through the integration harness, holding the
 * server requests of that first send.
 */
import {
  createFakeTldwServer,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  sendFromComposer,
  waitFor,
  waitForChatIdle,
  visibleNotifications,
  within,
  type FakeTldwServer,
  type PlaygroundView
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

const SELECTION_PATH = /^\/api\/v1\/chat\/conversations\/[^/]+\/history\/selection$/
const CREATE_PATH = /^\/api\/v1\/chats\/?$/
const LOADING = "Loading selected history"

/** Rows the transcript store holds, as role and text. */
const transcript = () =>
  useStoreMessageOption.getState().messages.map((message) => ({
    role: message.isBot ? "assistant" : "user",
    text: message.message
  }))

/** The rows the transcript renders, one entry per row, as their text. */
const renderedRows = () =>
  within(screen.getByRole("log", { name: "Chat messages" }))
    .queryAllByTestId("chat-message")
    .map((row) => row.textContent ?? "")

/** Assert the transcript renders exactly these rows, in order. */
const expectRendered = (texts: string[]) => {
  const rows = renderedRows()
  expect(rows).toHaveLength(texts.length)
  texts.forEach((text, index) => expect(rows[index]).toContain(text))
}

/** The message id of the rendered user row showing `text`. */
const questionRowId = (text: string) =>
  within(screen.getByRole("log", { name: "Chat messages" }))
    .queryAllByTestId("chat-message")
    .find((row) => row.getAttribute("data-role") === "user" && row.textContent?.includes(text))
    ?.getAttribute("data-message-id") ?? null

/** Every server chat's rows, as "role: content". */
const serverRows = (server: FakeTldwServer) =>
  [...server.chats.values()].map((chat) =>
    chat.messages.map((message) => `${message.role}: ${message.content}`)
  )

/** Type a message and press Enter without waiting for the turn. */
const pressEnterWith = async (view: PlaygroundView, text: string) => {
  await view.user.click(view.composer)
  await view.user.type(view.composer, text)
  await view.user.keyboard("{Enter}")
}

const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

describe("Playground first send in a new chat (CM-14)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("does not show 'Loading selected history' while the new chat's first load runs", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    // The first selection read is the new chat's load; hold it past the panel's delay.
    const load = server.holdRequest("POST", SELECTION_PATH)

    await pressEnterWith(view, "First question")
    await load.arrived
    await pause(400)
    expect(screen.queryByText(LOADING)).toBeNull()

    load.release()
    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "First question" },
          { role: "assistant", text: "Harness reply" }
        ]),
      HARNESS_WAIT
    )
    await waitForChatIdle()
    expect(screen.queryByText(LOADING)).toBeNull()
  })

  it("shows the question on Enter while the server creates the chat, then once", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    const create = server.holdRequest("POST", CREATE_PATH)

    try {
      await pressEnterWith(view, "First question")
      await create.arrived
      await waitFor(() => expectRendered(["First question"]), HARNESS_WAIT)
      expect(view.composer).toHaveValue("")
      expect(screen.queryByText(LOADING)).toBeNull()
    } finally {
      create.release()
    }

    await waitFor(
      () =>
        expect(transcript()).toEqual([
          { role: "user", text: "First question" },
          { role: "assistant", text: "Harness reply" }
        ]),
      HARNESS_WAIT
    )
    await waitForChatIdle()
    await waitFor(() => expectRendered(["First question", "Harness reply"]), HARNESS_WAIT)
    expect(serverRows(server)).toEqual([["user: First question", "assistant: Harness reply"]])
  })

  it("removes the question when the create fails and keeps its text only in the creation recovery", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    const create = server.holdRequest("POST", CREATE_PATH)

    try {
      await pressEnterWith(view, "Unsaved question")
      await create.arrived
      await waitFor(() => expectRendered(["Unsaved question"]), HARNESS_WAIT)
    } finally {
      create.fail(500, { detail: "Harness: create failed" })
    }

    await waitFor(
      () => expect(visibleNotifications().join(" ")).toContain("Server save unconfirmed"),
      HARNESS_WAIT
    )
    await waitForChatIdle()
    expect(transcript()).toEqual([])
    expectRendered([])
    expect(server.completionRequests()).toHaveLength(0)
    // The server may have created the chat, so the attempted text is kept in the
    // creation recovery rather than the composer (D7 P4): sending it again could
    // create a duplicate. It is shown there once, and nowhere else.
    expect(screen.getByText("Original attempted text")).toBeInTheDocument()
    expect(screen.getAllByText("Unsaved question")).toHaveLength(1)
  })

  it("shows a later question in the saved chat on Enter and never twice", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    await sendFromComposer(view, "First question")
    await waitFor(() => expectRendered(["First question", "Harness reply"]), HARNESS_WAIT)
    const capturesBefore = server.find("POST", SELECTION_PATH).length
    // The turn's own selection read comes before its rows are drawn.
    const turnCapture = server.holdRequest("POST", SELECTION_PATH)
    server.setDefaultReply("Second reply")

    let earlyId: string | null = null
    try {
      await pressEnterWith(view, "Second question")
      await turnCapture.arrived
      await waitFor(
        () => expectRendered(["First question", "Harness reply", "Second question"]),
        HARNESS_WAIT
      )
      earlyId = questionRowId("Second question")
      expect(earlyId).toBeTruthy()
    } finally {
      turnCapture.release()
    }

    const expected = [
      { role: "user", text: "First question" },
      { role: "assistant", text: "Harness reply" },
      { role: "user", text: "Second question" },
      { role: "assistant", text: "Second reply" }
    ]
    await waitFor(() => expect(transcript()).toEqual(expected), HARNESS_WAIT)
    await waitForChatIdle()
    // The view follows the settled reply with one more selection read.
    await waitFor(
      () => expect(server.find("POST", SELECTION_PATH).length).toBeGreaterThan(capturesBefore + 1),
      HARNESS_WAIT
    )
    await waitFor(() => expect(transcript()).toEqual(expected), HARNESS_WAIT)
    expectRendered(["First question", "Harness reply", "Second question", "Second reply"])
    // The turn drew its question under the id the early row used.
    expect(questionRowId("Second question")).toBe(earlyId)
  })

  it("leaves no stale question or history after a Stop right after the quiet load", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    // Let the new chat's load through and hold the turn's own selection read.
    const turnCapture = server.holdRequest("POST", SELECTION_PATH, { skip: 1 })

    try {
      await pressEnterWith(view, "Stopped question")
      await turnCapture.arrived
      await waitFor(() => expectRendered(["Stopped question"]), HARNESS_WAIT)
      await view.user.click(await screen.findByRole("button", { name: "Stop Streaming" }, HARNESS_WAIT))
    } finally {
      turnCapture.release()
    }

    await waitForChatIdle()
    await pause(100)
    expect(transcript()).toEqual([])
    expect(useStoreMessageOption.getState().history).toEqual([])
    expectRendered([])
    expect(screen.queryByText(LOADING)).toBeNull()
    expect(server.completionRequests()).toHaveLength(0)
    expect(serverRows(server)).toEqual([[]])
  })
})
