// @vitest-environment jsdom
/**
 * CM-05 (#3101), owner decision D4: Compare stays hidden in /chat until it is
 * verified. With the compare feature flag at its default (off), /chat must not
 * advertise compare: no empty-state starter card, and Alt+Shift+C is left
 * alone instead of being swallowed by a silent no-op. The Modes item and the
 * shortcuts help row are covered by their component tests.
 */
import {
  createFakeTldwServer,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  waitFor,
  within
} from "./harness/playground-harness"
import { fireEvent } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"

describe("Playground compare entry points while compare is unavailable", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("does not offer the compare starter card in the empty chat", async () => {
    const server = createFakeTldwServer()
    await renderPlayground({ server })

    const shell = await screen.findByTestId("playground-empty-shell", {}, HARNESS_WAIT)
    fireEvent.click(within(shell).getByRole("button", { name: "Explore chat modes" }))
    await waitFor(() =>
      expect(within(shell).getByTestId("playground-empty-mode-deck")).toBeInTheDocument()
    )

    expect(
      within(shell).queryByRole("button", { name: /Compare AI models side-by-side/i })
    ).toBeNull()
  })

  it("leaves Alt+Shift+C to the browser instead of swallowing it", async () => {
    const server = createFakeTldwServer()
    await renderPlayground({ server })

    const notPrevented = fireEvent.keyDown(document.body, {
      key: "C",
      code: "KeyC",
      altKey: true,
      shiftKey: true
    })

    expect(notPrevented).toBe(true)
  })
})
