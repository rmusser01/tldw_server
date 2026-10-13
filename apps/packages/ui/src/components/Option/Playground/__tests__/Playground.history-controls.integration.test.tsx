// @vitest-environment jsdom
/**
 * Red-first reproduction for #3106 (UX G05, CC-04): the history-path controls
 * ("Use no prior messages" / "Review conversation history") must not appear
 * for a fresh chat that has no history to review. The `it.fails` asserts the
 * correct behaviour and passes only while the defect reproduces.
 */
import {
  act,
  createFakeTldwServer,
  deferred,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  waitFor,
  waitForChatIdle
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"

describe("Playground history-path controls (#3106)", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  // CC-04 #3106 — HistorySelectionReview.tsx:326-333 renders "Use no prior messages" whenever the selection status is
  // "ready", even for the empty capture a fresh chat's first send installs (normalChatMode.ts:644-669).
  it.fails("CC-04 (#3106): a fresh chat with no reviewable history shows no history-path controls while its first message is sent", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    expect(screen.queryByRole("button", { name: "Use no prior messages" })).not.toBeInTheDocument()

    // Hold the reply so the first turn stays in flight while the controls are inspected.
    const reply = deferred()
    server.planCompletion({ gate: reply.promise })
    try {
      await view.user.click(view.composer)
      await view.user.type(view.composer, "Fresh question")
      await view.user.keyboard("{Enter}")
      await waitFor(() => expect(server.completionRequests()).toHaveLength(1), HARNESS_WAIT)
      await act(async () => {})

      expect(screen.queryByRole("button", { name: "Use no prior messages" })).not.toBeInTheDocument()
      expect(
        screen.queryByRole("button", { name: "Review conversation history" })
      ).not.toBeInTheDocument()
    } finally {
      reply.resolve()
      await waitForChatIdle()
    }
  })
})
