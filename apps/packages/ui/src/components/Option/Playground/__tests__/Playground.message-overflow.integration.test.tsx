// @vitest-environment jsdom
/**
 * CM-09 (#3101, G17): in a plain chat (no character or role-play scene) the
 * reply's More menu must not offer role-play steering such as "Continue as
 * user", "Impersonate user" or "Force narrate".
 */
import {
  createFakeTldwServer,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  sendFromComposer,
  within
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"

describe("Playground reply overflow menu", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("offers no role-play steering on a plain chat reply", async () => {
    const server = createFakeTldwServer()
    server.setDefaultReply("Plain answer")
    const view = await renderPlayground({ server })

    await sendFromComposer(view, "Plain question")

    await screen.findByText("Plain answer", {}, HARNESS_WAIT)
    const card = screen
      .getAllByTestId("chat-message")
      .find((row) => row.getAttribute("data-role") === "assistant")!
    await view.user.click(within(card).getByRole("button", { name: "More actions" }))
    await screen.findByText("Summarize this", {}, HARNESS_WAIT)

    for (const label of ["Continue as user", "Impersonate user", "Force narrate"]) {
      expect(screen.queryByText(label)).not.toBeInTheDocument()
    }
  })
})
