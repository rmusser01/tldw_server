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
  waitFor,
  waitForChatIdle,
  type PlaygroundView
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

const SELECTION_PATH = /^\/api\/v1\/chat\/conversations\/[^/]+\/history\/selection$/
const LOADING = "Loading selected history"

/** Rows the transcript store holds, as role and text. */
const transcript = () =>
  useStoreMessageOption.getState().messages.map((message) => ({
    role: message.isBot ? "assistant" : "user",
    text: message.message
  }))

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
})
