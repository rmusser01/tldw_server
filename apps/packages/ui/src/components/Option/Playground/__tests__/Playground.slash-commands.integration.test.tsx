// @vitest-environment jsdom
/**
 * CC-N1 (#3101): typing a slash command must not run it. Commands run only
 * when the user submits (Enter on the message, or picking the command in the
 * slash menu). Before the fix, `/web` flipped web search on almost every
 * keystroke and `/model` opened the settings dialog mid-word.
 */
import {
  completionMessages,
  createFakeTldwServer,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  sendFromComposer,
  waitFor,
  waitForChatIdle
} from "./harness/playground-harness"
import React from "react"
import { fireEvent } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"
import { isMac } from "@/hooks/useKeyboardShortcuts"
import { useStoreMessageOption } from "@/store/option"
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat"

const MODEL_SETTINGS_TITLE = "Current Chat Model Settings"
// antd ids are all "test-id" under NODE_ENV=test, so a dialog's
// aria-labelledby can resolve to another component; match its title text.
const modelSettingsDialog = () =>
  screen
    .queryAllByRole("dialog")
    .find((dialog) => dialog.textContent?.includes(MODEL_SETTINGS_TITLE)) ?? null

const modelSearch = () => screen.queryByRole("textbox", { name: "Search models" })

const typeOneKeyAtATime = async (
  view: Awaited<ReturnType<typeof renderPlayground>>,
  text: string,
  afterEachKey: (typedSoFar: string) => void
) => {
  let typed = ""
  for (const char of text) {
    await view.user.type(view.composer, char)
    typed += char
    afterEachKey(typed)
  }
}

// Opens a saved server chat the way the sidebar does, to check the draft it kept.
const sourceChat = { id: "" }
const ReopenSourceChat = () => {
  const select = useSelectServerChat()
  return (
    <button
      type="button"
      onClick={() =>
        select({ id: sourceChat.id, title: "First", state: "in-progress", version: 1, source: "webui-chat" } as Parameters<typeof select>[0])
      }>
      Reopen source chat
    </button>
  )
}

describe("Playground slash commands", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("leaves web search alone while '/web …' is typed and toggles it once on send", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    expect(useStoreMessageOption.getState().webSearch).toBe(false)

    await view.user.click(view.composer)
    const seen: string[] = []
    await typeOneKeyAtATime(view, "/web abc", (typed) => {
      if (useStoreMessageOption.getState().webSearch) seen.push(typed)
    })
    expect(seen).toEqual([])
    expect(view.composer).toHaveValue("/web abc")

    await view.user.keyboard("{Enter}")
    await waitFor(
      () => expect(useStoreMessageOption.getState().webSearch).toBe(true),
      HARNESS_WAIT
    )
    await waitForChatIdle()
    expect(useStoreMessageOption.getState().webSearch).toBe(true)
    // The text after the command is sent as the message, with web search on.
    expect(server.find("POST", "/api/v1/research/websearch")).toHaveLength(1)
    const completions = server.completionRequests()
    expect(completions).toHaveLength(1)
    expect(completionMessages(completions[0]).at(-1)).toMatchObject({ role: "user" })
    expect(completionMessages(completions[0]).at(-1)?.content).toContain("abc")
  })

  it("keeps web search unchanged when a typed '/web' is deleted without sending", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })

    await view.user.click(view.composer)
    await view.user.type(view.composer, "/web")
    await view.user.clear(view.composer)

    expect(useStoreMessageOption.getState().webSearch).toBe(false)
  })

  // Decision [4]: on the web '@' attaches notes, so the hint names it (CC-11
  // kept it out while '@' meant browser tabs, which only the extension reads).
  it("names @ mentions in the web app's composer, where they attach notes", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })

    expect(view.composer.getAttribute("placeholder")).not.toContain("@")
    expect(screen.getByText("/ commands · @ mentions")).toBeInTheDocument()
  })

  // Decision [4]: '@' picks a note; its content goes out with the message as
  // pinned context does, and it stays as a removable chip linking to the note.
  it("attaches a note picked with @ and sends its content with the message", async () => {
    const server = createFakeTldwServer({
      notes: [{ id: 7, title: "Rebase notes", content: "Use git rebase -i to squash." }]
    })
    const view = await renderPlayground({ server })
    await view.user.click(view.composer)
    await view.user.type(view.composer, "@Reb")
    await screen.findByRole("option", { name: /Rebase notes/ }, HARNESS_WAIT)
    await view.user.keyboard("{Enter}")
    expect(view.composer).toHaveValue("")
    expect(screen.getByRole("link", { name: "Rebase notes" })).toHaveAttribute("href", "/notes?note=7")
    await view.user.type(view.composer, "Summarize this")
    await view.user.keyboard("{Enter}")
    await waitForChatIdle()
    const completions = server.completionRequests()
    expect(completions).toHaveLength(1)
    const sent = String(completionMessages(completions[0]).at(-1)?.content)
    expect(sent).toContain("Summarize this")
    expect(sent).toContain("Use git rebase -i to squash.")
  })

  // Review fix: an attached note belongs to one message. After it is sent the
  // chip clears, and the next message goes without it.
  it("sends an attached note with one message only", async () => {
    const server = createFakeTldwServer({
      notes: [{ id: 7, title: "Rebase notes", content: "Use git rebase -i to squash." }]
    })
    const view = await renderPlayground({ server })
    await view.user.click(view.composer)
    await view.user.type(view.composer, "@Reb")
    await screen.findByRole("option", { name: /Rebase notes/ }, HARNESS_WAIT)
    await view.user.keyboard("{Enter}")
    await screen.findByRole("link", { name: "Rebase notes" }, HARNESS_WAIT)
    await view.user.type(view.composer, "First question")
    await view.user.keyboard("{Enter}")
    await waitForChatIdle()
    expect(screen.queryByRole("link", { name: "Rebase notes" })).toBeNull()
    await view.user.type(view.composer, "Second question")
    await view.user.keyboard("{Enter}")
    await waitForChatIdle()
    const completions = server.completionRequests()
    expect(completions).toHaveLength(2)
    expect(String(completionMessages(completions[0]).at(-1)?.content)).toContain("Use git rebase -i to squash.")
    expect(String(completionMessages(completions[1]).at(-1)?.content)).toContain("Second question")
    expect(String(completionMessages(completions[1]).at(-1)?.content)).not.toContain("Use git rebase -i to squash.")
  })

  // Review fix: a note is an explicit attachment, so file retrieval does not drop it.
  it("sends an attached note with file retrieval on", async () => {
    const server = createFakeTldwServer({
      notes: [{ id: 7, title: "Rebase notes", content: "Use git rebase -i to squash." }]
    })
    const view = await renderPlayground({ server })
    useStoreMessageOption.getState().setFileRetrievalEnabled(true)
    await view.user.click(view.composer)
    await view.user.type(view.composer, "@Reb")
    await screen.findByRole("option", { name: /Rebase notes/ }, HARNESS_WAIT)
    await view.user.keyboard("{Enter}")
    await view.user.type(view.composer, "Summarize this")
    await view.user.keyboard("{Enter}")
    await waitForChatIdle()
    const completions = server.completionRequests()
    expect(completions).toHaveLength(1)
    expect(String(completionMessages(completions[0]).at(-1)?.content)).toContain("Use git rebase -i to squash.")
  })

  // AX-01: '/model' opens the composer's model picker (search focused), not
  // the chat settings, and only once the command is chosen.
  it("opens the model picker only when '/model' is chosen, not while it is typed", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })

    await view.user.click(view.composer)
    const openedAt: string[] = []
    await typeOneKeyAtATime(view, "/model", (typed) => {
      if (modelSettingsDialog() || modelSearch()) openedAt.push(typed)
    })
    // The settings dialog is lazy-loaded; give it time to appear if it was opened.
    await new Promise((resolve) => setTimeout(resolve, 1_500))
    if (modelSettingsDialog() || modelSearch()) openedAt.push("after typing")
    expect(openedAt).toEqual([])
    expect(view.composer).toHaveFocus()

    await view.user.keyboard("{Enter}")
    await waitFor(() => expect(modelSearch()).toHaveFocus(), HARNESS_WAIT)
    expect(screen.getByTestId("model-selector")).toHaveAttribute("aria-expanded", "true")
    expect(modelSettingsDialog()).toBeNull()
  })

  it("opens the model picker from the composer with the Switch model shortcut", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })

    await view.user.click(view.composer)
    fireEvent.keyDown(view.composer, { key: "e", metaKey: isMac, ctrlKey: !isMac })

    await waitFor(() => expect(modelSearch()).toHaveFocus(), HARNESS_WAIT)
    expect(modelSettingsDialog()).toBeNull()
  })

  // CC-11 (slice a): /new, /prompt and the server's chat commands.
  it("starts a fresh saved chat with '/new', like the header's New chat", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    await sendFromComposer(view, "First topic question")
    const previousHistoryId = useStoreMessageOption.getState().historyId

    await view.user.click(view.composer)
    await view.user.type(view.composer, "/new")
    await view.user.keyboard("{Enter}")

    await waitFor(
      () => expect(screen.queryByText("First topic question")).not.toBeInTheDocument(),
      HARNESS_WAIT
    )
    expect(view.composer).toHaveValue("")
    expect(useStoreMessageOption.getState()).toMatchObject({ temporaryChat: false, messages: [] })
    await sendFromComposer(view, "Unrelated question")
    expect(useStoreMessageOption.getState().historyId).not.toBe(previousHistoryId)
    expect(completionMessages(server.completionRequests().at(-1)!)).toEqual([
      { role: "user", content: "Unrelated question" }
    ])
  })

  // CC-11 review: '/new' must not leave its own text behind as the draft of
  // the chat it leaves; text typed after it moves to the new chat.
  it.each([
    ["picked from the menu", "/new", ""],
    ["submitted with text after it", "/new and then this", "and then this"]
  ])("leaves no command draft in the chat '/new' leaves (%s)", async (_how, typed, carried) => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server, extras: <ReopenSourceChat /> })
    await sendFromComposer(view, "First topic question")
    sourceChat.id = String(useStoreMessageOption.getState().serverChatId)

    await view.user.click(view.composer)
    await view.user.type(view.composer, typed)
    await view.user.keyboard("{Enter}")

    await waitFor(
      () => expect(screen.queryByText("First topic question")).not.toBeInTheDocument(),
      HARNESS_WAIT
    )
    await waitForChatIdle()
    expect(server.completionRequests()).toHaveLength(1)
    expect(useStoreMessageOption.getState().messages).toEqual([])
    await waitFor(() => expect(screen.getByTestId("chat-input")).toHaveValue(carried), HARNESS_WAIT)

    // Let the new chat's draft save, then go back to the chat it left.
    await new Promise((resolve) => setTimeout(resolve, 1_200))
    await view.user.click(screen.getByRole("button", { name: "Reopen source chat" }))
    await screen.findByText("First topic question", {}, HARNESS_WAIT)
    await waitForChatIdle()
    await new Promise((resolve) => setTimeout(resolve, 1_200))
    expect(screen.getByTestId("chat-input")).toHaveValue("")
  })

  it("opens the prompt picker with '/prompt'", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })

    await view.user.click(view.composer)
    await view.user.type(view.composer, "/prompt")
    await view.user.keyboard("{Enter}")

    expect(await screen.findByPlaceholderText("Search prompts...", {}, HARNESS_WAIT)).toBeInTheDocument()
    expect(view.composer).toHaveValue("")
    expect(server.completionRequests()).toHaveLength(0)
  })

  it("searches the prompt picker for the text after '/prompt' and sends nothing", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })

    await view.user.click(view.composer)
    await view.user.type(view.composer, "/prompt weekly review")
    await view.user.keyboard("{Enter}")

    expect(await screen.findByPlaceholderText("Search prompts...", {}, HARNESS_WAIT)).toHaveValue("weekly review")
    await waitForChatIdle()
    expect(server.completionRequests()).toHaveLength(0)
    expect(view.composer).toHaveValue("")
  })

  it("lists the server's chat commands and sends a picked one as typed", async () => {
    const server = createFakeTldwServer({
      chatCommands: [{ name: "time", description: "Show the current time" }]
    })
    const view = await renderPlayground({ server })

    await view.user.click(view.composer)
    await view.user.type(view.composer, "/ti")
    expect(await screen.findByText("Show the current time", {}, HARNESS_WAIT)).toBeInTheDocument()
    await view.user.keyboard("{Enter}")
    await waitFor(() => expect(view.composer).toHaveValue("/time "), HARNESS_WAIT)

    await view.user.type(view.composer, "UTC")
    await view.user.keyboard("{Enter}")
    await waitForChatIdle()
    const completions = server.completionRequests()
    expect(completions).toHaveLength(1)
    expect(completionMessages(completions[0]).at(-1)).toEqual({ role: "user", content: "/time UTC" })
  })

  it("keeps the local commands when the server lists none, and asks only once '/' is typed", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    // Loading /chat does not ask for the server's commands.
    expect(server.find("GET", "/api/v1/chat/commands")).toHaveLength(0)

    await view.user.click(view.composer)
    await view.user.type(view.composer, "/")
    await waitFor(() => expect(server.find("GET", "/api/v1/chat/commands")).toHaveLength(1), HARNESS_WAIT)
    expect(screen.getByText("/new")).toBeInTheDocument()
    expect(screen.getByText("/prompt")).toBeInTheDocument()
    expect(screen.queryByText("/time")).toBeNull()
    expect(server.unhandled).toEqual([])
  })
})
