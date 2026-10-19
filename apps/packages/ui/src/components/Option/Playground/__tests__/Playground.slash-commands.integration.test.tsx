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
  waitFor,
  waitForChatIdle
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

const MODEL_SETTINGS_TITLE = "Current Chat Model Settings"
const modelSettingsDialog = () =>
  screen.queryByRole("dialog", { name: MODEL_SETTINGS_TITLE })

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

  it("does not open model settings while '/model' is typed, only when the command is chosen", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })

    await view.user.click(view.composer)
    const openedAt: string[] = []
    await typeOneKeyAtATime(view, "/model", (typed) => {
      if (modelSettingsDialog()) openedAt.push(typed)
    })
    // The settings dialog is lazy-loaded; give it time to appear if it was opened.
    await new Promise((resolve) => setTimeout(resolve, 1_500))
    if (modelSettingsDialog()) openedAt.push("after typing")
    expect(openedAt).toEqual([])
    expect(view.composer).toHaveFocus()

    await view.user.keyboard("{Enter}")
    await waitFor(() => expect(modelSettingsDialog()).not.toBeNull(), HARNESS_WAIT)
  })
})
