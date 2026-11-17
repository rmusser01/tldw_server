// @vitest-environment jsdom
/**
 * CO-07 (#3101, G12): the floating "Scroll to latest messages" button must
 * not cover an empty chat.
 */
import {
  createFakeTldwServer,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"

describe("Playground scroll to latest button", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("is not shown on an empty chat", async () => {
    const server = createFakeTldwServer()
    await renderPlayground({ server })
    await screen.findByRole("textbox", {}, HARNESS_WAIT).catch(() => null)

    expect(
      screen.queryByRole("button", { name: /scroll to latest messages/i })
    ).not.toBeInTheDocument()
  })
})
