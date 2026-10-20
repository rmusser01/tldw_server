// @vitest-environment jsdom
/**
 * Cut-off replies on /chat (#3104 follow-up, UX G06: CM-03).
 *
 * A reply that ends because the model hit its output limit
 * (`finish_reason: "length"`), or because the connection closed before the
 * server's final `[DONE]`, used to look like a finished answer. It must be
 * marked as incomplete and offer Continue and a retry. These run against the
 * real Playground through the integration harness.
 */
import {
  createFakeTldwServer,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  waitFor,
  waitForChatIdle,
  within,
  type PlaygroundView
} from "./harness/playground-harness"
import { act } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

const sendInTemporaryChat = async (view: PlaygroundView, text: string) => {
  act(() => useStoreMessageOption.getState().startNewChatDraft(true))
  const before = view.server.completionRequests().length
  await view.user.click(view.composer)
  await view.user.type(view.composer, text)
  await view.user.keyboard("{Enter}")
  await waitFor(() => expect(view.server.completionRequests()).toHaveLength(before + 1), HARNESS_WAIT)
  await waitForChatIdle()
}

const lastReply = () => {
  const rows = screen.getAllByTestId("chat-message").filter((row) => row.getAttribute("data-role") === "assistant")
  return rows.at(-1)!
}

describe("Playground cut-off replies (CM-03)", { timeout: 90_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("CM-03: a reply whose stream closes without [DONE] keeps the text, is marked Interrupted, and offers Continue and Retry", async () => {
    const server = createFakeTldwServer()
    server.planCompletion({
      chunks: ["Here is the first part of a long answer", " that was cut off mid-"],
      end: "eof"
    })
    const view = await renderPlayground({ server })

    await sendInTemporaryChat(view, "Explain everything")

    const marker = await within(lastReply()).findByText("Interrupted", { selector: "[data-interruption-label]" }, HARNESS_WAIT)
    const notice = marker.closest("[role='status']") as HTMLElement
    // The reply text renders once the lazy Markdown view has loaded.
    await waitFor(
      () => expect(lastReply()).toHaveTextContent("Here is the first part of a long answer that was cut off mid-"),
      HARNESS_WAIT
    )
    expect(within(notice).getByRole("button", { name: "Retry same model" })).toBeEnabled()
    expect(within(notice).getByRole("button", { name: "Continue from partial" })).toBeEnabled()
    const reply = useStoreMessageOption.getState().messages.at(-1)!
    expect(reply.generationInfo).toMatchObject({ interrupted: true })
  })

  it("CM-03: a reply that finished normally is not marked incomplete", async () => {
    const server = createFakeTldwServer()
    server.planCompletion({ chunks: ["A complete", " answer."] })
    const view = await renderPlayground({ server })

    await sendInTemporaryChat(view, "Explain briefly")

    await waitFor(() => expect(lastReply()).toHaveTextContent("A complete answer."), HARNESS_WAIT)
    expect(within(lastReply()).queryByRole("status", { name: "Output limit reached" })).toBeNull()
    expect(within(lastReply()).queryByText("Interrupted")).toBeNull()
  })
})
