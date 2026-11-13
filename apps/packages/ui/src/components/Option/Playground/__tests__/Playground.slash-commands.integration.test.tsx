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
import { fireEvent } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"
import { isMac } from "@/hooks/useKeyboardShortcuts"
import { useStoreMessageOption } from "@/store/option"

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

  // CC-11: '@' mentions are browser tabs, which only the extension can read.
  it("does not promise @ mentions in the web app's composer", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })

    expect(view.composer.getAttribute("placeholder")).not.toContain("@")
    expect(screen.getByText("/ commands")).toBeInTheDocument()
    expect(screen.queryByText(/@ mentions/)).toBeNull()
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

  it("sends nothing when '/new' is submitted with text after it", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    await sendFromComposer(view, "First topic question")

    await view.user.click(view.composer)
    await view.user.type(view.composer, "/new and then this")
    await view.user.keyboard("{Enter}")

    await waitFor(
      () => expect(screen.queryByText("First topic question")).not.toBeInTheDocument(),
      HARNESS_WAIT
    )
    await waitForChatIdle()
    expect(server.completionRequests()).toHaveLength(1)
    expect(useStoreMessageOption.getState().messages).toEqual([])
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

  it("keeps the local commands when the server lists none", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })

    await view.user.click(view.composer)
    await view.user.type(view.composer, "/")
    await waitFor(() => expect(server.find("GET", "/api/v1/chat/commands")).toHaveLength(1), HARNESS_WAIT)
    expect(screen.getByText("/new")).toBeInTheDocument()
    expect(screen.getByText("/prompt")).toBeInTheDocument()
    expect(screen.queryByText("/time")).toBeNull()
    expect(server.unhandled).toEqual([])
  })
})
