// @vitest-environment jsdom
/**
 * CM-09 (#3101, G17): in a plain chat (no character or role-play scene) the
 * reply's More menu must not offer role-play steering such as "Continue as
 * user", "Impersonate user" or "Force narrate".
 *
 * Steering is a page-wide intent in the option store, applied by every send.
 * Because a plain chat hides the steering controls, an intent armed in a
 * character chat must not follow the user into a plain chat, where it would
 * shape the next reply with no way to see or clear it.
 */
import {
  act,
  completionMessages,
  createFakeTldwServer,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  sendFromComposer,
  waitFor,
  within
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"
import { DEFAULT_MESSAGE_STEERING_PROMPTS } from "@/utils/message-steering"

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

  it("does not carry steering armed in a character chat into a plain chat", async () => {
    const server = createFakeTldwServer()
    server.setDefaultReply("Plain answer")
    const view = await renderPlayground({ server })
    await sendFromComposer(view, "Plain question")
    await screen.findByText("Plain answer", {}, HARNESS_WAIT)

    // The state a character chat leaves behind when the user armed "Force
    // narrate" or "Impersonate user" and then switched here without sending:
    // nothing on a chat switch resets the page-wide steering intent.
    act(() => {
      const state = useStoreMessageOption.getState()
      state.setMessageSteeringMode("impersonate_user")
      state.setMessageSteeringForceNarrate(true)
    })

    await waitFor(() => {
      const state = useStoreMessageOption.getState()
      expect({
        mode: state.messageSteeringMode,
        forceNarrate: state.messageSteeringForceNarrate
      }).toEqual({ mode: "none", forceNarrate: false })
    }, HARNESS_WAIT)

    await sendFromComposer(view, "Second plain question")

    const sent = completionMessages(server.completionRequests().at(-1)!)
    expect(sent.at(-1)).toEqual({ role: "user", content: "Second plain question" })
    const steeringPrompts = [
      DEFAULT_MESSAGE_STEERING_PROMPTS.forceNarrate,
      DEFAULT_MESSAGE_STEERING_PROMPTS.impersonateUser,
      DEFAULT_MESSAGE_STEERING_PROMPTS.continueAsUser
    ]
    for (const message of sent) {
      for (const prompt of steeringPrompts) {
        expect(message.content).not.toContain(prompt)
      }
    }
  })
})
