// @vitest-environment jsdom
/**
 * Cut-off replies on /chat (#3104 follow-up, UX G06: CM-03, and Continue: CM-01).
 *
 * A reply that ends because the model hit its output limit
 * (`finish_reason: "length"`), or because the connection closed before the
 * server's final `[DONE]`, used to look like a finished answer. It must be
 * marked as incomplete and offer Continue and a retry. These run against the
 * real Playground through the integration harness.
 */
import {
  completionMessages,
  createFakeTldwServer,
  HARNESS_WAIT,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  visibleNotifications,
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
  it("CM-01: Continue on a cut-off reply in a temporary chat sends the conversation and finishes the reply", async () => {
    const server = createFakeTldwServer()
    server.planCompletion({ chunks: ["Here is the first part", " of a long answer"], end: "eof" })
    server.planCompletion({ reply: " and the rest of it." })
    const view = await renderPlayground({ server })

    await sendInTemporaryChat(view, "Explain everything")
    const marker = await within(lastReply()).findByText("Interrupted", { selector: "[data-interruption-label]" }, HARNESS_WAIT)
    const notice = marker.closest("[role='status']") as HTMLElement

    await view.user.click(within(notice).getByRole("button", { name: "Continue from partial" }))
    await waitFor(() => expect(server.completionRequests()).toHaveLength(2), HARNESS_WAIT)
    await waitForChatIdle()

    // The conversation goes out with the cut-off reply last, so the model picks
    // up where it stopped. It used to go out empty and fail with
    // "Stream completion failed".
    expect(completionMessages(server.completionRequests()[1])).toEqual([
      { role: "user", content: "Explain everything" },
      { role: "assistant", content: "Here is the first part of a long answer" }
    ])
    expect(visibleNotifications()).toEqual([])
    await waitFor(
      () => expect(lastReply()).toHaveTextContent("Here is the first part of a long answer and the rest of it."),
      HARNESS_WAIT
    )
    expect(within(lastReply()).queryByText("Interrupted")).toBeNull()
  })

  it("CM-03: a reply that hit the output limit is marked incomplete and offers Continue and Retry", async () => {
    const server = createFakeTldwServer()
    server.planCompletion({ chunks: ["Here is the first part of a long answer that was cut off mid-"], finishReason: "length" })
    server.planCompletion({ reply: " sentence, and the rest." })
    const view = await renderPlayground({ server })

    await sendInTemporaryChat(view, "Explain everything")

    const notice = await within(lastReply()).findByRole("status", { name: "Output limit reached" }, HARNESS_WAIT)
    expect(notice).toHaveTextContent(/output limit/i)
    expect(within(notice).getByRole("button", { name: "Retry same model" })).toBeEnabled()
    const continueButton = within(notice).getByRole("button", { name: "Continue from partial" })

    await view.user.click(continueButton)
    await waitFor(() => expect(server.completionRequests()).toHaveLength(2), HARNESS_WAIT)
    await waitForChatIdle()
    // Continue sends the conversation with the cut-off reply last.
    expect(completionMessages(server.completionRequests()[1])).toEqual([
      { role: "user", content: "Explain everything" },
      { role: "assistant", content: "Here is the first part of a long answer that was cut off mid-" }
    ])
    await waitFor(
      () => expect(lastReply()).toHaveTextContent("Here is the first part of a long answer that was cut off mid- sentence, and the rest."),
      HARNESS_WAIT
    )
    // The continued reply finished normally, so it is no longer marked incomplete.
    expect(within(lastReply()).queryByRole("status", { name: "Output limit reached" })).toBeNull()
  })
})
